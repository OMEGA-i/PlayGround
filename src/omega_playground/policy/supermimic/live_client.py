"""Own the resident Joystick subprocess without mixing the two MuJoCo installations."""
import os
import socket
import subprocess
from multiprocessing.connection import Connection
from pathlib import Path


class JoystickTimeout(TimeoutError):
    """The Joystick failed to return its reference within the request deadline."""


class JoystickClient:
    def __init__(self,robot,xml):
        root=Path(__file__).resolve().parents[4]
        joystick=root.parent/'Joystick'
        parent,child=socket.socketpair()
        try:
            self.proc=subprocess.Popen([os.environ.get('OMEGA_JOYSTICK_PYTHON', str(joystick/'.venv/bin/python')),str(root/'tools/supermimic/supermimic_joystick_worker.py'),
                                        str(child.fileno()),robot,str(xml)],pass_fds=(child.fileno(),),
                                       cwd=joystick,env={**os.environ,'PYTHONPATH':str(root/'src')})
        finally:
            child.close()
        self.link=Connection(parent.detach())
        try:
            ready = self._receive(180)
            self.provenance = ready.get('provenance')
            if not ready.get('ready'):
                raise RuntimeError('Joystick failed to initialize')
        except BaseException:
            self.close()
            raise

    def _receive(self,timeout):
        if not self.link.poll(timeout):
            raise JoystickTimeout(f'Joystick did not respond within {timeout} s')
        result=self.link.recv()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def generate(self,vx,vy,reset=False,observation=None):
        self.link.send({'velocity':[vx,vy],'reset':reset,'observation':observation})
        return self._receive(20)

    def generate_many(self, requests):
        """One resident model, isolated rolling contexts for independent worlds."""
        self.link.send({'batch': requests})
        return self._receive(max(20, 20 * len(requests)))['references']

    def close(self):
        if self.proc.poll() is None:
            try:
                self.link.send(None)
                self.proc.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        self.link.close()
