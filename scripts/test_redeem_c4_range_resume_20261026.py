"""Offline regression tests, no Zenodo network access and no data downloads."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest

S = Path(__file__).with_name('redeem_c4_range_resume_20261026.py')
spec = importlib.util.spec_from_file_location('c4_range', S)
c4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c4)

class Server(BaseHTTPRequestHandler):
    payload = b''
    return_full = False
    corrupt = False
    def log_message(self, *args):
        pass
    def do_GET(self):
        data = type(self).payload
        import re
        m = re.fullmatch(r'bytes=(\d+)-(\d+)', self.headers.get('Range',''))
        if self.return_full or not m:
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            try: self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError): pass
            return
        lo, hi = map(int, m.groups())
        if lo >= len(data) or hi >= len(data):
            self.send_response(416)
            self.end_headers()
            return
        out = data[lo:hi+1]
        if self.corrupt and lo != 0:
            out = b'x' + out[1:]
        self.send_response(206)
        self.send_header('Content-Length', str(len(out)))
        self.send_header('Content-Range', f'bytes {lo}-{hi}/{len(data)}')
        self.end_headers()
        try: self.wfile.write(out)
        except (BrokenPipeError, ConnectionResetError): pass

class Cases(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        Server.payload = (bytes(range(256)) * 17017)
        Server.return_full = False
        Server.corrupt = False
        self.srv = ThreadingHTTPServer(('127.0.0.1', 0), Server)
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)
        self.url = f'http://127.0.0.1:{self.srv.server_port}/zip'
        self.part = self.dir/c4.PART_NAME
        self.final = self.dir/c4.ZIP_NAME
        self.md5 = hashlib.md5(Server.payload, usedforsecurity=False).hexdigest()
    def test_download_full_and_publisher_digest(self):
        c4.recover(self.url, self.part, self.final, 'resume', 2, self.md5, len(Server.payload))
        self.assertEqual(self.final.read_bytes(), Server.payload)
        self.assertFalse(self.part.exists())
        self.assertTrue((self.dir/'nine-species-balanced.recovery-provenance.json').exists())
    def test_preserve_partial_and_resume(self):
        self.part.write_bytes(Server.payload[:20147])
        c4.recover(self.url, self.part, self.final, 'probe', 2, self.md5, len(Server.payload))
        self.assertEqual(self.part.stat().st_size, 20147)
        c4.recover(self.url, self.part, self.final, 'resume', 2, self.md5, len(Server.payload))
        self.assertEqual(self.final.read_bytes(), Server.payload)
    def test_reject_non_range_http_200(self):
        self.part.write_bytes(Server.payload[:100])
        Server.return_full = True
        with self.assertRaises(c4.SourceChanged):
            c4.recover(self.url, self.part, self.final, 'resume', 2, self.md5, len(Server.payload))
        self.assertEqual(self.part.stat().st_size, 100)
        self.assertFalse(self.final.exists())
    def test_reject_changed_partial_before_append(self):
        self.part.write_bytes(b'X' + Server.payload[1:2000])
        with self.assertRaises(c4.SourceChanged):
            c4.recover(self.url, self.part, self.final, 'resume', 2, self.md5, len(Server.payload))
        self.assertEqual(self.part.stat().st_size, 2000)
        self.assertFalse(self.final.exists())
    def test_reject_changed_full_md5(self):
        with self.assertRaises(c4.SourceChanged):
            c4.recover(self.url, self.part, self.final, 'resume', 2, '0'*32, len(Server.payload))
        self.assertTrue(self.part.exists())
        self.assertFalse(self.final.exists())
    def test_refuse_overwrite(self):
        self.final.write_bytes(b'protected')
        with self.assertRaises(FileExistsError):
            c4.recover(self.url, self.part, self.final, 'resume', 2, self.md5, len(Server.payload))
        self.assertEqual(self.final.read_bytes(), b'protected')

if __name__ == '__main__': unittest.main()
