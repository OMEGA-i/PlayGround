import json
import socket
from types import SimpleNamespace

from wsproto import ConnectionType, WSConnection
from wsproto.events import AcceptConnection, BytesMessage, Request, TextMessage

from omega_playground.ops.server import OperationsServer
from omega_playground.ops.video import FragmentedVideo


def test_authenticated_ws_sends_metadata_init_and_acknowledged_fragments(tmp_path):
    server = OperationsServer(SimpleNamespace(meta={}), 'scene', 0)
    video = FragmentedVideo(tmp_path/'unused')
    video.init = b'init'
    video.sequence = 1
    video.fragments.append((1, b'fragment'))
    server.start()
    server.video = video
    port = server.http.server_port
    client = WSConnection(ConnectionType.CLIENT)
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=2) as sock:
            sock.sendall(client.send(Request(host=f'127.0.0.1:{port}', target='/api/video-ws',
                          extra_headers=[(b'origin', f'http://127.0.0.1:{port}'.encode())],
                          subprotocols=['omega-video', 'token.'+server.token])))
            accepted, metadata, packets = False, None, []
            while len(packets) < 2:
                client.receive_data(sock.recv(65536))
                for event in client.events():
                    if isinstance(event, AcceptConnection):
                        accepted = True
                    elif isinstance(event, TextMessage):
                        metadata = json.loads(event.data)
                    elif isinstance(event, BytesMessage):
                        packets.append(event.data)
                        sock.sendall(client.send(TextMessage(data='ack')))
            assert accepted and metadata['mime'].startswith('video/mp4')
            assert packets == [b'init', b'fragment']
    finally:
        server.close()


def test_ws_without_token_is_rejected(tmp_path):
    import urllib.error
    import urllib.request

    import pytest
    server = OperationsServer(SimpleNamespace(meta={}), 'scene', 0)
    server.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f'http://127.0.0.1:{server.http.server_port}/api/video-ws')
        assert error.value.code == 403
    finally:
        server.close()
