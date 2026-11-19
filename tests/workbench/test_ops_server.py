import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from omega_playground.ops.server import OperationsServer


def test_http_commands_require_token_local_host_and_origin(tmp_path):
    nav = SimpleNamespace(meta={'bounds': [0, 0, 10, 10]}, directory=tmp_path)
    server = OperationsServer(nav, 'scene', port=0)
    server.start()
    base = f'http://127.0.0.1:{server.http.server_port}'

    def post(headers):
        req = urllib.request.Request(base + '/api/command', b'{"id":"qc","kind":"pause"}', headers)
        return urllib.request.urlopen(req)

    try:
        with pytest.raises(urllib.error.HTTPError) as missing:
            post({})
        assert missing.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as origin:
            post({'X-Omega-Token': server.token, 'Origin': 'http://elsewhere.test'})
        assert origin.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as host:
            post({'X-Omega-Token': server.token, 'Host': 'elsewhere.test'})
        assert host.value.code == 403
        assert server.commands.empty()
        with post({'X-Omega-Token': server.token, 'Origin': base}) as result:
            assert result.status == 202
            assert json.load(result) == {'id': 'qc', 'queued': True}
        assert server.commands.get_nowait()['kind'] == 'pause'
    finally:
        server.close()


def test_stream_publishes_complete_latest_frames_and_closes(tmp_path):
    server = OperationsServer(SimpleNamespace(meta={}, directory=tmp_path), 'scene', port=0)
    server.start()
    server.publish_frame(b'first-frame')
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{server.http.server_port}/api/stream', timeout=2) as stream:
            assert stream.headers['Content-Type'].startswith('multipart/x-mixed-replace')
            for frame in (b'first-frame', b'next-frame'):
                assert stream.readline() == b'--omega-frame\r\n'
                assert stream.readline() == b'Content-Type: image/jpeg\r\n'
                assert stream.readline() == b'Content-Length: ' + str(len(frame)).encode() + b'\r\n'
                assert stream.readline() == b'\r\n'
                assert stream.read(len(frame)+2) == frame+b'\r\n'
                if frame == b'first-frame':
                    server.publish_frame(b'next-frame')
    finally:
        server.close()




def test_workflow_capabilities_and_immediate_camera_endpoint(tmp_path):
    server = OperationsServer(SimpleNamespace(meta={}, directory=tmp_path), 'scene', port=0)
    server.start()
    base = f'http://127.0.0.1:{server.http.server_port}'
    try:
        with urllib.request.urlopen(base + '/api/bootstrap') as response:
            assert json.load(response)['task_start'] is True
        with urllib.request.urlopen(base + '/workflow.js') as response:
            assert response.headers.get_content_type() == 'text/javascript'
            assert b'OmegaWorkflow' in response.read()
        request = urllib.request.Request(base + '/api/viewer',
                    json.dumps({'id': 'camera-qc', 'camera': 'qc', 'agent': 0}).encode(),
                    {'X-Omega-Token': server.token, 'Origin': base})
        with urllib.request.urlopen(request) as response:
            assert response.status == 202
        command = server.commands.get_nowait()
        assert command['kind'] == 'viewer' and command['camera'] == 'qc'
    finally:
        server.close()
