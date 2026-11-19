"""Authenticated binary fMP4 over WebSocket, with a bounded acknowledged fragment window."""
import json
import select
import time

from wsproto import ConnectionType, WSConnection
from wsproto.events import AcceptConnection, BytesMessage, CloseConnection, Ping, Request, TextMessage
from wsproto.utilities import ProtocolError


def stream(handler, owner):
    protocols = [p.strip() for p in handler.headers.get('Sec-WebSocket-Protocol', '').split(',')]
    if 'omega-video' not in protocols or 'token.' + owner.token not in protocols:
        return handler.reply({'error': 'video token required'}, 403)
    origin = handler.headers.get('Origin')
    if origin != 'http://' + handler.headers.get('Host', ''):
        return handler.reply({'error': 'same origin required'}, 403)
    video = owner.video
    if video is None:
        return handler.reply({'error': 'video disabled'}, 404)
    ws = WSConnection(ConnectionType.SERVER)
    request = f'GET {handler.path} HTTP/1.1\r\n' + ''.join(f'{k}: {v}\r\n' for k,v in handler.headers.items()) + '\r\n'
    try:
        ws.receive_data(request.encode('latin1'))
        if not any(isinstance(event, Request) for event in ws.events()):
            return handler.reply({'error': 'invalid websocket upgrade'}, 400)
        handler.connection.sendall(ws.send(AcceptConnection(subprotocol='omega-video')))
        handler.close_connection = True
        handler.connection.settimeout(.5)

        def send(event):
            handler.connection.sendall(ws.send(event))

        send(TextMessage(data=json.dumps({'mime': owner.video_mime})))
        with video.condition:
            video.condition.wait_for(lambda: video.closed or (video.init and video.fragments), timeout=15)
            if video.closed or not video.init or not video.fragments:
                send(CloseConnection(code=1013, reason='Waiting for video'))
                return
            init = video.init
            sequence = video.fragments[-1][0] - 1
        send(BytesMessage(data=init))
        inflight, message, last_ack = 1, '', time.monotonic()
        while not owner.closed:
            if select.select([handler.connection], [], [], .02)[0]:
                data = handler.connection.recv(4096)
                if not data:
                    return
                ws.receive_data(data)
                for event in ws.events():
                    if isinstance(event, CloseConnection):
                        send(event.response())
                        return
                    if isinstance(event, Ping):
                        send(event.response())
                    if isinstance(event, TextMessage):
                        message += event.data
                        if len(message) > 32:
                            return
                        if event.message_finished:
                            if message != 'ack' or inflight <= 0:
                                return
                            inflight -= 1
                            message, last_ack = '', time.monotonic()
            if inflight and time.monotonic() - last_ack > 8:
                send(CloseConnection(code=1013, reason='Video acknowledgements timed out'))
                return
            with video.condition:
                if video.closed:
                    send(CloseConnection(code=1000, reason='Video ended'))
                    return
                if video.fragments and sequence < video.fragments[0][0] - 1:
                    send(CloseConnection(code=1013, reason='Slow video consumer'))
                    return
                chunks = [(seq, data) for seq, data in video.fragments if seq > sequence][:12-inflight]
            for sequence, data in chunks:
                send(BytesMessage(data=data))
                inflight += 1
    except (OSError, ValueError, ProtocolError):
        return
