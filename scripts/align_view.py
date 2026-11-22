"""Live alignment page: camera view blended with the demo start frame (green edges = demo object positions). http://127.0.0.1:8894"""
import urllib.request, numpy as np, cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
demo = cv2.imread("results/photos/demo_ep14_start.jpg"); edges = cv2.Canny(cv2.cvtColor(demo, cv2.COLOR_BGR2GRAY), 60, 150) > 0
PAGE = b"""<html><body style="background:#111;margin:0"><img id=i style="width:100%;max-width:1280px"><script>setInterval(()=>{i.src='/o.jpg?'+Date.now()},300)</script></body></html>"""
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/o.jpg"):
            try:
                live = cv2.imdecode(np.frombuffer(urllib.request.urlopen("http://127.0.0.1:8893/frame.jpg?c=table", timeout=3).read(), np.uint8), 1)
                ov = cv2.addWeighted(live, 0.65, demo, 0.35, 0); ov[edges] = (0, 255, 0); cv2.putText(ov, "move bowl/spoon/cup onto the GREEN outlines", (8, 22), 0, 0.6, (0, 0, 255), 2)
                b = cv2.imencode(".jpg", cv2.resize(ov, (1280, 960)))[1].tobytes(); self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers(); self.wfile.write(b)
            except Exception: self.send_response(503); self.end_headers()
        else: self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(PAGE)
ThreadingHTTPServer(("127.0.0.1", 8894), H).serve_forever()
