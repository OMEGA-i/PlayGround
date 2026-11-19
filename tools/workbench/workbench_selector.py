"""Loopback control for nginx: expose one shared active physics world to every browser."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import hmac
import json
import os
import threading
from urllib.request import urlopen

PORTS={'sonic':18782,'supermimic':18783}
selected=os.environ.get('OMEGA_WORKFLOW', 'sonic')
if selected not in PORTS:
    raise ValueError('OMEGA_WORKFLOW must be sonic or supermimic')
lock=threading.Lock()

def fetch(policy,path):
    with urlopen(f'http://127.0.0.1:{PORTS[policy]}/api/{path}',timeout=3) as r:
        return json.load(r)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_):pass

    def reply(self,status,body):
        data=json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Content-Type','application/json')
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Omega-Port',str(PORTS[selected]))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path!='/active':return self.reply(404,{})
        with lock:self.reply(200,dict(workflow=selected))

    def do_POST(self):
        global selected
        if self.path!='/switch':return self.reply(404,{})
        with lock:
            try:
                n=int(self.headers.get('Content-Length',0))
                if not 0<n<=128:raise ValueError
                target=json.loads(self.rfile.read(n))['workflow']
                if target not in PORTS:raise ValueError
                if not hmac.compare_digest(self.headers.get('X-Omega-Token',''),fetch(selected,'bootstrap')['token']):
                    return self.reply(403,dict(error='页面已过期，请刷新'))
                if not fetch(selected,'state').get('paused'):
                    return self.reply(409,dict(error='先暂停当前仿真，再切换策略'))
                if not fetch(target,'state').get('ready'):
                    return self.reply(503,dict(error='目标策略尚未就绪'))
                selected=target
                self.reply(200,dict(workflow=target))
            except (ValueError,KeyError,TypeError):
                self.reply(400,dict(error='Invalid workflow'))
            except OSError:
                self.reply(503,dict(error='目标物理服务不可用'))


if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',18784),Handler).serve_forever()
