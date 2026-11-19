"""Spawned renderer: independent Python interpreter and CUDA context."""

import multiprocessing as mp
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full
from threading import Condition, Thread


def mjwarp_scene_viewer(*args, **kwargs):
    # Import only inside the render worker. Isaac Sim and MJWarp have distinct
    # Warp environments and must not load each other's renderer dependencies.
    from omega_playground.viewer.mjwarp_live import mjwarp_viewer
    return mjwarp_viewer(*args, **kwargs)


def _render(factory, requests, responses):
    viewer = None
    cuda_slots = {}
    cuda_frames = 0
    frozen_gc = False
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='frame-encode') as encoding:
            viewer = factory()
            if getattr(viewer, 'freeze_static_gc', False):
                # This dedicated process owns one scene for its whole lifetime.
                # Exclude long-lived scene/kernel objects from periodic full scans;
                # leave collection of newly allocated frame objects enabled.
                import gc
                gc.collect()
                gc.freeze()
                frozen_gc = True
            responses.put(('ready',))
            pending = deque()

            def deliver(rgb, record, token, sim_time, frames, ego):
                try:
                    jpeg = viewer.encode_frame(rgb, record)
                    responses.put(('frame', token, jpeg, sim_time, record, frames, time.perf_counter(), ego))
                except BaseException:
                    responses.put(('error', traceback.format_exc()))
                    raise

            while True:
                item = requests.get()
                if item is None:
                    break
                token, poses, sim_time, record, settings = item
                if isinstance(poses, dict) and 'cuda_slot' in poses:
                    slot = poses['cuda_slot']
                    if 'storage' in poses:
                        rebuild, metadata = poses['storage']
                        cuda_slots[slot] = rebuild(*metadata)
                    poses = cuda_slots[slot]
                    cuda_frames += 1
                for name, value in settings.items():
                    setattr(viewer, name, value)
                if getattr(viewer, 'stream_video', False):
                    # Two owned frames maximum: preserve order and backpressure.
                    if len(pending) >= 2:
                        pending.popleft().result()
                    def submit(rgb, record, token=token, sim_time=sim_time, frames=viewer.frames):
                        pending.append(encoding.submit(deliver, rgb, record, token, sim_time, frames+int(record), getattr(viewer, 'ego_jpeg', None)))
                    viewer.frame_sink = submit
                    viewer.show(poses, sim_time, record=record)
                else:
                    viewer.show(poses, sim_time, record=record)
                    responses.put(('frame', token, viewer.latest_jpeg, sim_time, record,
                                   viewer.frames, time.perf_counter()))
            for future in pending:
                future.result()
            result = viewer.close()
            if cuda_slots:
                import torch
                torch.cuda.synchronize()
                poses = item = None
                cuda_slots.clear()
            if isinstance(result, dict):
                result['cuda_pose_frames'] = cuda_frames
            viewer = None
            responses.put(('closed', result))
    except BaseException:  # noqa: BLE001 -- propagate worker termination to the owner
        responses.put(('error', traceback.format_exc()))
    finally:
        if viewer is not None:
            viewer.close()
        if frozen_gc:
            gc.unfreeze()


class RenderProcess:
    def __init__(self, factory, on_frame, height_query=None, python_executable=None):
        self._height_query = height_query
        context = mp.get_context('spawn')  # Never fork a live CUDA context.
        self._requests, self._responses = context.Queue(2), context.Queue(2)
        self._condition = Condition()
        self._ready = self._closed = False
        self._error = self.result = None
        self._token = self._completed = 0
        self._cuda_slots = {}
        self.camera_mode, self.follow_body, self.root_bodies = 'qc', 0, [0]
        self.orbit_yaw, self.orbit_pitch, self.orbit_distance = 30., -12., 4.
        self.latest_jpeg, self.frames = None, 0
        self.latest_sim_time = None
        if python_executable is None:
            self._process = context.Process(target=_render, args=(factory, self._requests, self._responses),
                                            name='omega-render', daemon=True)
            self._process.start()
        else:
            from omega_playground.viewer.worker import ExternalRenderer
            self._process = ExternalRenderer(factory, self._requests, self._responses, python_executable)
        self._listener = Thread(target=self._listen, args=(on_frame,), daemon=True)
        self._listener.start()
        self._wait(lambda: self._ready)

    def _check(self):
        if self._error:
            raise RuntimeError('Rendering process failed:\n' + self._error)
        if self._process.exitcode is not None and not self._closed:
            raise RuntimeError(f'Rendering process exited: {self._process.exitcode}')

    def _wait(self, predicate):
        with self._condition:
            while not predicate():
                self._check()
                self._condition.wait(.1)
            self._check()

    def _listen(self, on_frame):
        try:
            while True:
                try:
                    item = self._responses.get(timeout=.1)
                except Empty:
                    self._check()
                    continue
                with self._condition:
                    if item[0] == 'ready':
                        self._ready = True
                    elif item[0] == 'frame':
                        _, token, jpeg, sim_time, record, frames, wall, *ego = item
                        self.ego_jpeg = ego[0] if ego else None
                        self.latest_jpeg, self.frames = jpeg, frames
                        self.latest_sim_time = sim_time
                        on_frame(jpeg, sim_time, record, wall)
                        self._completed = token
                    elif item[0] == 'closed':
                        self.result, self._closed = item[1], True
                    elif item[0] == 'error':
                        self._error = item[1]
                    self._condition.notify_all()
                    if self._closed or self._error:
                        return
        except BaseException:  # noqa: BLE001 -- wake waiters instead of silently losing the listener
            with self._condition:
                self._error = traceback.format_exc()
                self._condition.notify_all()

    def _put(self, item):
        while True:
            self._check()
            try:
                self._requests.put(item, timeout=.1)
                return
            except Full:
                pass

    def show(self, body_q, sim_time, record=True):
        settings = {name: getattr(self, name) for name in (
            'camera_mode', 'follow_body', 'root_bodies', 'orbit_yaw', 'orbit_pitch', 'orbit_distance')}
        if self.camera_mode == 'qc' and self._height_query is not None:
            root = body_q[self.follow_body, :3]
            if getattr(root, 'is_cuda', False):
                root = root.detach().cpu().numpy()
            settings['camera_ground_z'] = self._height_query(root)
        self._token += 1
        if getattr(body_q, 'is_cuda', False):
            import torch
            from torch.multiprocessing.reductions import reduce_tensor
            slot_id = self._token % 2
            previous = self._cuda_slots.get(slot_id)
            if previous is not None:
                buffer, last_token = previous
                self._wait(lambda: self._completed >= last_token)
                if buffer.shape != body_q.shape or buffer.dtype != body_q.dtype or buffer.device != body_q.device:
                    raise ValueError('CUDA render topology changed; rebuild the renderer')
            else:
                buffer = torch.empty_like(body_q)
            buffer.copy_(body_q)
            torch.cuda.current_stream(body_q.device).synchronize()
            payload = {'cuda_slot': slot_id}
            if previous is None:
                payload['storage'] = reduce_tensor(buffer)
            self._cuda_slots[slot_id] = buffer, self._token
        else:
            payload = body_q.copy()
        self._put((self._token, payload, sim_time, record, settings))
        if not record:
            self._wait(lambda: self._completed >= self._token)

    def close(self):
        try:
            self._put(None)
            self._wait(lambda: self._closed)
            self._process.join(timeout=10)
            return self.result
        finally:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=10)
            self._listener.join(timeout=10)
            for queue in (self._requests, self._responses):
                queue.close()
                if self._closed:
                    queue.join_thread()
                else:
                    # A dead consumer may leave a feeder blocked on its pipe.
                    queue.cancel_join_thread()
            # Kit can exit before Python's normal finalizer sweep. Release the
            # queue-owned semaphore objects while this interpreter is still up.
            self._requests = self._responses = None
            self._cuda_slots.clear()
