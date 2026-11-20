from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from threading import Event, Thread
from time import monotonic, sleep
import unittest

import cv2
import numpy as np

from localization.camera_stream import XiaoCameraSource, mjpeg_parts


class CameraStreamTest(unittest.TestCase):
    def test_quoted_boundary_and_optional_length(self):
        body = BytesIO(b"--cam\r\nContent-Type: image/jpeg\r\nContent-Length: 3\r\n"
                       b"\r\na\nb\r\n--cam\r\ncontent-type: image/jpeg\r\n\r\n"
                       b"jpeg-data\r\n--cam--\r\n")
        parts = list(mjpeg_parts('multipart/x-mixed-replace; boundary="cam"', body))
        self.assertEqual([payload for _, payload in parts], [b"a\nb", b"jpeg-data"])

    def test_bad_length_is_rejected(self):
        body = BytesIO(b"--cam\r\nContent-Type: image/jpeg\r\nContent-Length: -2\r\n\r\n")
        with self.assertRaises(ValueError):
            next(mjpeg_parts("multipart/x-mixed-replace; boundary=cam", body))

    def test_reconnect_decode_timestamps_and_latest_buffer(self):
        jpeg = cv2.imencode(".jpg", np.full((32, 48, 3), (20, 80, 180), np.uint8))[1].tobytes()
        first_consumed = Event()

        class Server(BaseHTTPRequestHandler):
            requests = 0

            def log_message(self, *_):
                pass

            def do_GET(self):
                Server.requests += 1
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=cam")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                ids = (1, 2, 4) if Server.requests == 1 else (5, 6)
                try:
                    for index in ids:
                        data = b"bad jpeg" if index == 2 else jpeg
                        part = (f"--cam\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n"
                                f"X-Frame-Id: {index}\r\nX-Stream-Session: a\r\n\r\n".encode()
                                + data + b"\r\n")
                        self.wfile.write(f"{len(part):X}\r\n".encode() + part + b"\r\n")
                        self.wfile.flush()
                        if index == 1:
                            first_consumed.wait(1)
                        sleep(0.05)
                    if Server.requests > 1:
                        sleep(0.5)
                    self.wfile.write(b"0\r\n\r\n")
                except OSError:
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        source = XiaoCameraSource(f"http://127.0.0.1:{server.server_port}/stream",
                                  timeout_s=0.5, retry_s=0.06)
        try:
            source.start()
            first = source.read(timeout=1)
            self.assertEqual(first.frame_id, 1)
            self.assertEqual(first.image.shape, (32, 48, 3))
            first_consumed.set()
            until = monotonic() + 2
            while source.status().received < 4 and monotonic() < until:
                sleep(0.005)
            latest = source.read(timeout=0.2)
            self.assertEqual(latest.frame_id, 4)
            self.assertGreater(latest.timestamp, first.timestamp)
            self.assertEqual(latest.sequence, 6)
            stats = source.status()
            self.assertEqual(stats.source_gaps, 1)
            self.assertEqual(stats.decode_errors, 1)
            self.assertGreaterEqual(stats.overwritten, 1)
            self.assertGreaterEqual(stats.reconnections, 1)
        finally:
            first_consumed.set()
            source.close()
            server.shutdown()
            server.server_close()
            worker.join()
        self.assertFalse(source.running)


if __name__ == "__main__":
    unittest.main()
