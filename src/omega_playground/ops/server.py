"""Loopback-only HTTP surface; simulation owns all mutations via a bounded command queue."""

import json
import math
import queue
import secrets
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


class OperationsServer:
    def __init__(self, navigation, scene_hash, port=8765, video_path=None, video_mime='video/mp4; codecs="avc1.42E01F"'):
        from omega_playground.ops.video import FragmentedVideo
        self.video = FragmentedVideo(video_path) if video_path is not None else None
        self.video_mime = video_mime
        self.navigation, self.scene_hash = navigation, scene_hash
        self.token = secrets.token_urlsafe(32)
        self.session_id = secrets.token_hex(16)
        self.commands = queue.Queue(maxsize=32)
        self.events = deque(maxlen=256)
        self.snapshot = {"ready": False, "paused": True}
        self.viewer_frame = None
        self.viewer_jpeg = None
        self.frame_condition = threading.Condition()
        self.frame_sequence = 0
        self.closed = False
        self.lock = threading.Lock()
        owner = self
        self._preview_navigation = None
        self._occupancy_png = None

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, value, status=200, content_type="application/json"):
                data = json.dumps(finite_json(value), allow_nan=False).encode() if content_type == "application/json" else value
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(data)

            def valid_host(self):
                host = self.headers.get('Host', '')
                return host.split(':')[0] in ('127.0.0.1', 'localhost')

            def do_GET(self):
                if not self.valid_host():
                    return self.reply({'error': 'local Host required'}, 403)
                path = urlsplit(self.path).path
                if path == '/api/bootstrap':
                    return self.reply({'token': owner.token, 'session_id': owner.session_id, 'scene_sha256': owner.scene_hash,
                                       'scene': getattr(owner,'active_scene',None), 'scenes': getattr(owner,'scenes',[]),
                                       'prepared_options': getattr(owner, 'prepared_options', None),
                                       'video_transport': 'websocket', 'video_mime': owner.video_mime,
                                       'map': owner.navigation.meta, 'locomotion_modes': ['walk', 'run'], 'task_start': True})
                if path == '/api/state':
                    with owner.lock:
                        data = owner.snapshot
                    return self.reply(data)
                if path == '/api/events':
                    with owner.lock:
                        events = list(owner.events)
                    return self.reply(events)
                if path == '/occupancy.png':
                    from omega_playground.map.preview import occupancy_png
                    if owner._occupancy_png is None:
                        owner._occupancy_png = occupancy_png(owner.preview_navigation())
                    return self.reply(owner._occupancy_png, content_type='image/png')
                if path == '/map.png':
                    return self.reply((owner.navigation.directory / 'campus.png').read_bytes(), content_type='image/png')
                if path == '/api/video-ws':
                    from omega_playground.ops.websocket_video import stream
                    return stream(self, owner)
                if path == '/api/video':
                    video = owner.video
                    if video is None:
                        return self.reply({'error': 'video disabled'}, 404)
                    with video.condition:
                        if not video.init or not video.fragments:
                            return self.reply({'error': 'waiting for recorded frames'}, 503)
                        init = video.init
                        sequence = video.fragments[-1][0] - 1
                    self.send_response(200)
                    self.send_header('Content-Type', 'video/mp4')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.connection.settimeout(5)
                    try:
                        self.wfile.write(init)
                        while True:
                            with video.condition:
                                video.condition.wait_for(
                                    lambda sequence=sequence: video.closed or video.sequence > sequence, timeout=1)
                                if video.closed:
                                    return
                                chunks = [(seq, data) for seq, data in video.fragments if seq > sequence]
                            for sequence, data in chunks:
                                self.wfile.write(data)
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        return
                if path in ('/api/stream','/api/ego-stream'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=omega-frame')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.connection.settimeout(5)
                    sequence = -1
                    try:
                        while True:
                            with owner.frame_condition:
                                owner.frame_condition.wait_for(
                                    lambda sequence=sequence: owner.closed or owner.frame_sequence != sequence, timeout=1)
                                if owner.closed:
                                    return
                                if owner.frame_sequence == sequence:
                                    continue
                                sequence = owner.frame_sequence
                                frame = getattr(owner, "ego_jpeg", None) if path == "/api/ego-stream" else owner.viewer_jpeg
                            if frame is None:
                                continue
                            self.wfile.write(b'--omega-frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                             + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        return
                if path == '/api/ego-viewer':
                    frame = getattr(owner, 'ego_jpeg', None)
                    if frame is None:
                        return self.reply({'error': 'waiting for ego frame'}, 503)
                    return self.reply(frame, content_type='image/jpeg')
                if path == '/api/viewer' and owner.viewer_jpeg is not None:
                    return self.reply(owner.viewer_jpeg, content_type='image/jpeg')
                if path == '/api/viewer' and owner.viewer_frame is not None:
                    return self.reply(owner.viewer_frame.read_bytes(), content_type='image/png')
                assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                          '/video.js': ('video.js', 'text/javascript; charset=utf-8'),
                          '/workflow.js': ('workflow.js', 'text/javascript; charset=utf-8'),
                          '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if path in assets:
                    filename, kind = assets[path]
                    return self.reply((Path(__file__).parent / 'web' / filename).read_bytes(), content_type=kind)
                return self.reply({'error': 'not found'}, 404)

            def do_POST(self):
                origin = self.headers.get('Origin')
                if (not self.valid_host() or self.headers.get('X-Omega-Token') != owner.token
                        or (origin is not None and origin != 'http://' + self.headers.get('Host', ''))):
                    return self.reply({'error': 'token and same origin required'}, 403)
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 65536:
                        raise ValueError('body must contain 1..65536 bytes')
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise TypeError('JSON object required')
                    if self.path == '/api/drawn-path':
                        from omega_playground.control.drawn_path import drawn_path
                        from omega_playground.map.preview import inspect_path
                        plan = drawn_path(body.get('keypoints'))
                        plan['validation'] = inspect_path(owner.preview_navigation(), plan['route'], plan['clearance_m'])
                        return self.reply(plan)
                    if self.path == '/api/viewer':
                        body['kind'] = 'viewer'
                    if self.path in ('/api/command', '/api/viewer'):
                        allowed = ('viewer',) if self.path == '/api/viewer' else ('pause', 'resume', 'reset', 'select_scene', 'control_mode', 'keyboard', 'drawn_path', 'locomotion', 'configure_robot')
                        if body.get('kind') not in allowed:
                            raise ValueError('unknown command')
                        if not isinstance(body.get('id'), str) or not 1 <= len(body['id']) <= 128:
                            raise ValueError('command id required, at most 128 characters')
                        owner.commands.put_nowait(body)
                        return self.reply({'id': body['id'], 'queued': True}, 202)
                    return self.reply({'error': 'not found'}, 404)
                except queue.Full:
                    return self.reply({'error': 'command queue full'}, 429)
                except (ValueError, KeyError, TypeError) as exc:
                    return self.reply({'error': str(exc)}, 400)

        self.http = self.thread = None
        if port is None:
            return  # Offline runtime retains commands/events without opening a socket.
        self.http = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    def preview_navigation(self):
        if self._preview_navigation is None:
            from omega_playground.map.navigation import CampusNavigation
            self._preview_navigation = (self.navigation if hasattr(self.navigation, 'clearance') else
                                        CampusNavigation(self.navigation.directory, self.scene_hash))
        return self._preview_navigation

    def start(self):
        if self.video is not None:
            self.video.thread.start()
        if self.thread is not None:
            self.thread.start()

    def publish(self, snapshot):
        with self.lock:
            self.snapshot = snapshot

    def publish_frame(self, jpeg):
        with self.frame_condition:
            self.viewer_jpeg = jpeg
            self.frame_sequence += 1
            self.frame_condition.notify_all()

    def event(self, event):
        with self.lock:
            self.events.append(event)

    def close(self):
        if self.video is not None:
            self.video.close()
        with self.frame_condition:
            self.closed = True
            self.frame_condition.notify_all()
        if self.http is not None:
            self.http.shutdown()
            self.http.server_close()
            self.thread.join(timeout=2)
