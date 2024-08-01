"""
Use threading to capture video. Always get the latest video frame to avoid buffer cache delay
"""

import cv2
import queue
import threading

# Class for reading video streams without buffering
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 90)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # Immediately read a frame when available, only keep the latest frame
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # Discard the previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

def main():
    # Test code
    cap = VideoCapture(0)
    while True:
        frame = cap.read()
        cv2.imshow("kmboxNvideo-thread", frame)
        if chr(cv2.waitKey(1) & 255) == 'q':
            break

if __name__ == "__main__":
    main()
