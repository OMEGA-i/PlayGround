"""External-interpreter renderer transport; never inherits the simulator's sys.path."""
import multiprocessing.connection
import socket
import subprocess
import sys
import traceback
from pathlib import Path
from threading import Lock, Thread


class ConnectionQueue:
    def __init__(self, connection):
        self.connection, self.lock = connection, Lock()

    def get(self):
        return self.connection.recv()

    def put(self, item):
        with self.lock:
            self.connection.send(item)


class ExternalRenderer:
    def __init__(self, factory, requests, responses, python):
        executable = Path(python).absolute()
        if not executable.is_file():
            raise ValueError(f'Renderer Python does not exist: {executable}')
        parent, child = socket.socketpair()
        self.connection = multiprocessing.connection.Connection(parent.detach())
        try:
            self.process = subprocess.Popen([str(executable), '-m', 'omega_playground.viewer.worker', str(child.fileno())],
                                            pass_fds=(child.fileno(),))
        finally:
            child.close()
        self.connection.send(factory)
        def send():
            try:
                while True:
                    item = requests.get()
                    self.connection.send(item)
                    if item is None:
                        return
            except (EOFError, OSError):
                pass  # Reader/lifecycle reports worker failure to the owner.
        def receive():
            try:
                while True:
                    item = self.connection.recv()
                    responses.put(item)
                    if item[0] in ('closed', 'error'):
                        return
            except (EOFError, OSError):
                responses.put(('error', 'External renderer disconnected\n'+traceback.format_exc()))
        Thread(target=send, daemon=True, name='render-send').start()
        Thread(target=receive, daemon=True, name='render-receive').start()

    @property
    def exitcode(self):
        return self.process.poll()

    def is_alive(self):
        return self.process.poll() is None

    def join(self, timeout=None):
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return
        self.connection.close()

    def terminate(self):
        self.process.terminate()


if __name__ == '__main__':
    from omega_playground.viewer.process import _render
    connection = multiprocessing.connection.Connection(int(sys.argv[1]))
    try:
        factory = connection.recv()
        queue = ConnectionQueue(connection)
        _render(factory, queue, queue)
    finally:
        connection.close()
