"""Bounded live view of the recorder's fragmented MP4; no second encoder."""

import threading
from collections import deque


class FragmentedVideo:
    def __init__(self, path):
        self.path = path
        self.condition = threading.Condition()
        self.closed = False
        self.init = b''
        self.fragments = deque(maxlen=10)
        self.sequence = 0
        self._buffer = b''
        self._prefix = b''
        self._fragment = b''
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True, name='omega-video-tail')

    def feed(self, data):
        self._buffer += data
        while len(self._buffer) >= 8:
            size = int.from_bytes(self._buffer[:4], 'big')
            kind = self._buffer[4:8]
            if size == 1:
                if len(self._buffer) < 16:
                    return
                size = int.from_bytes(self._buffer[8:16], 'big')
            if not 8 <= size <= 64 * 1024 * 1024:
                raise ValueError('Invalid MP4 box size')
            if len(self._buffer) < size:
                return
            box, self._buffer = self._buffer[:size], self._buffer[size:]
            with self.condition:
                if kind == b'ftyp':
                    self._prefix = box
                elif kind == b'moov':
                    self.init = self._prefix + box
                elif kind == b'moof':
                    self._fragment = box
                elif kind == b'mdat' and self._fragment:
                    self.sequence += 1
                    self.fragments.append((self.sequence, self._fragment + box))
                    self._fragment = b''
                self.condition.notify_all()

    def _watch(self):
        try:
            while not self.path.exists():
                if self._stop.wait(.05):
                    return
            with self.path.open('rb') as file:
                while not self._stop.is_set():
                    data = file.read(1024 * 1024)
                    if data:
                        self.feed(data)
                    else:
                        self._stop.wait(.02)
        finally:
            with self.condition:
                self.closed = True
                self.condition.notify_all()

    def close(self):
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
