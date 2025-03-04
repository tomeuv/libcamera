#!/usr/bin/env python3

from __future__ import annotations

import argparse
import cv2
import kms
import libcamera as libcam
import mmap
import selectors
import sys
import time
import numpy as np
import os
from pixutils import dmaheap

from collections import deque

from MappedFrameBuffer import MappedFrameBuffer

from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

def time_ms():
    return time.time() * 1000


class FPSCounter:
    def __init__(self, name=""):
        self.start_time = None
        self.frame_count = 0
        self.name = name
        self.fps = 0

    def tick(self):
        if self.start_time is None:
            self.start_time = time.time()
            self.frame_count = 0

        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time >= 2:
            self.fps = self.frame_count / elapsed_time
            #print(f"{self.name} FPS: {self.fps:.2f}")
            self.start_time = current_time
            self.frame_count = 0


class MyBuf:
    idx: int
    cam: libcam.Camera
    stream: libcam.Stream
    fb: kms.DumbFramebuffer
    buffer: libcam.FrameBuffer


class KMSState:
    def __init__(self, mybufs: list[MyBuf], queue_buf, heap):
        self.fps = FPSCounter("KMS")

        self.mybufs = mybufs
        self.queue_buf = queue_buf

        card = kms.Card()
        res = kms.ResourceManager(card)
        conn = res.reserve_connector()
        crtc = res.reserve_crtc(conn)
        mode = conn.get_default_mode()

        self.card = card
        self.crtc = crtc
        self.conn = conn
        self.mode = mode

        req = kms.AtomicReq(card)
        req.add(self.crtc, 'ACTIVE', 0)
        req.commit_sync(allow_modeset = True)

        stream = mybufs[0].stream

        cfg = stream.configuration
        size = cfg.size
        w = size.width
        h = size.height
        fmt = kms.PixelFormats.RGB888

        for mybuf in mybufs:
            buf = heap.alloc(fmt.framesize(w, h))
            fb = kms.DmabufFramebuffer(card, w, h,
                                       fmt,
                                       fds=[ buf.fd ],
                                       pitches=[ fmt.stride(w) ],
                                       offsets=[ 0 ])
            mybuf.fb = fb
            mybuf.heap_buf = buf

        self.in_queue = deque()
        # Committed fb
        self.next_fb = None
        # On screen fb
        self.current_fb = None
        # Previous fb, will be unused on next pageflip
        self.prev_fb = None

    def setup(self, mybuf: MyBuf):
        # Do a modeset with the given buffer to get the display up
        fb = mybuf.fb
        kms.AtomicReq.set_mode(self.conn, self.crtc, fb, self.mode)
        self.current_fb = mybuf

    def queue_new_frame(self, mybuf: MyBuf):
        self.in_queue.append(mybuf)
        if not self.next_fb:
            self.handle_page_flip()

    def handle_page_flip(self):
        self.fps.tick()

        assert self.current_fb

        if self.prev_fb:
            self.queue_buf(self.prev_fb)
            self.prev_fb = None

        # Did we have something committed? If so, it's now current
        if self.next_fb:
            self.prev_fb = self.current_fb
            self.current_fb = self.next_fb
            self.next_fb = None

        if len(self.in_queue) > 0:
            self.next_fb = self.in_queue.popleft()

            ctx = kms.AtomicReq(self.card)
            ctx.add(self.crtc.primary_plane, "FB_ID", self.next_fb.fb.id)
            ctx.commit()

    def readdrm(self):
        for ev in self.card.read_events():
            if ev.type == kms.DrmEventType.FLIP_COMPLETE:
                self.handle_page_flip()


class CamState:
    def __init__(
        self, camera_id: int | str, format_str: str | None, size_str: str | None
    ):
        self.cam_fps = FPSCounter("Cam")

        self.cm = libcam.CameraManager.singleton()

        try:
            if camera_id.isnumeric():
                cam_idx = int(camera_id)
                cam = next(
                    (cam for i, cam in enumerate(self.cm.cameras) if i + 1 == cam_idx)
                )
            else:
                cam = next((cam for cam in self.cm.cameras if camera_id in cam.id))
        except Exception:
            print(f'Failed to find camera "{camera_id}"')
            return -1

        cam.acquire()

        cam_config = cam.generate_configuration([libcam.StreamRole.Viewfinder])

        stream_config = cam_config.at(0)

        if format_str:
            fmt = libcam.PixelFormat(format_str)
            stream_config.pixel_format = fmt

        if size_str:
            w, h = [int(v) for v in size_str.split("x")]
            stream_config.size = libcam.Size(w, h)

        cam_config.validate()

        cam.configure(cam_config)

        print(f"Camera configured to {stream_config}")

        self.cam = cam
        self.stream_config = stream_config
        self.stream = stream_config.stream

        self.is_first_req = True

        model_dir = os.environ['MODEL_PATH']
        model_path = os.path.join(model_dir, 'detect.tflite')
        labels_path = os.path.join(model_dir, 'labelmap.txt')

        self.interpreter = Interpreter(model_path=model_path,
                                       experimental_delegates=[load_delegate(os.environ['DELEGATE_PATH'])])
        self.interpreter.allocate_tensors()

        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def add_req(self, mybuf: MyBuf):
        # Use the buffer index as the cookie
        req = self.cam.create_request(mybuf.idx)

        req.add_buffer(self.stream, mybuf.buffer.fb)

        if self.is_first_req:
            # Looks like on rpi5 we get "interesting" fps if we don't set it explicitly
            req.set_control(libcam.controls.FrameDurationLimits, (33333, 33333))
            self.is_first_req = False

        self.cam.queue_request(req)

    def setup_hack(self, mybufs: list[MyBuf], kmsstate: KMSState):
        self.mybufs = mybufs
        self.kmsstate = kmsstate

    def infer_and_draw(self, frame):
        input_details = self.interpreter.get_input_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        frame_resized = cv2.resize(frame, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Retrieve detection results
        output_details = self.interpreter.get_output_details()
        boxes = self.interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = self.interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

        imH = 1080
        imW = 1920

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if (scores[i] <= 0.55) or (scores[i] > 1.0):
                continue

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(self.kmsstate.fps.fps),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        return frame

    def handle_req(self):
        self.cam_fps.tick()

        reqs = self.cm.get_ready_requests()

        for req in reqs:
            buffers = req.buffers

            assert len(buffers) == 1

            idx = req.cookie
            mybuf = self.mybufs[idx]

            kmsbuf = np.frombuffer(mybuf.fb.map(0), dtype=np.uint8)  # assume single plane

            w = self.stream.configuration.size.width
            h = self.stream.configuration.size.height
            cambuf = np.frombuffer(mybuf.buffer.planes[0], dtype=np.uint8)
            cambuf = cambuf.reshape((h, w, 2))
            kmsbuf = kmsbuf.reshape((h, w, 3))

            dmaheap.sync_start(mybuf.fb.planes[0].prime_fd, write=1)

            dmaheap.sync_start(mybuf.buffer.fb.planes[0].fd)
            cv2.cvtColor(cambuf, cv2.COLOR_YUV2RGB_YVYU, kmsbuf)
            dmaheap.sync_end(mybuf.buffer.fb.planes[0].fd)

            self.infer_and_draw(kmsbuf)

            dmaheap.sync_end(mybuf.fb.planes[0].prime_fd, write=1)

            self.kmsstate.queue_new_frame(mybuf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--camera",
        type=str,
        default="1",
        help="Camera index number (starting from 1) or part of the name",
    )
    parser.add_argument("-f", "--format", type=str, help="Pixel format")
    parser.add_argument("-s", "--size", type=str, help='Size ("WxH")')
    args = parser.parse_args()

    # Camera setup
    camstate = CamState(args.camera, args.format, args.size)

    # Allocate framebuffers

    stream = camstate.stream

    allocator = libcam.FrameBufferAllocator(camstate.cam)
    ret = allocator.allocate(stream)
    assert ret > 0

    num_bufs = stream.configuration.buffer_count

    mybufs = []
    heap = dmaheap.DMAHeap('reserved')
    for i in range(num_bufs):
        buf_size = stream.configuration.size.width * stream.configuration.size.height * 2
        buf = heap.alloc(buf_size)
        plane = libcam.FrameBuffer.Plane(buf.fd, 0, buf_size)
        fb = libcam.FrameBuffer([plane])

        mybuf = MyBuf()
        mybuf.idx = i
        mybuf.cam = camstate.cam
        mybuf.stream = stream
        mybuf.buffer = MappedFrameBuffer(fb)
        mybuf.fb = None
        mybufs.append(mybuf)

        mybuf.buffer.mmap()

    kmsstate = KMSState(mybufs, camstate.add_req, heap)
    # Give the first buffer to kms
    kmsstate.setup(mybufs[0])

    camstate.setup_hack(mybufs, kmsstate)

    # Start camera. Need to start it before we can queue buffers
    camstate.cam.start()

    # skip the first buf, as it has been given to kms
    for i in range(1, num_bufs):
        camstate.add_req(mybufs[i])

    def handle_key_event():
        sys.stdin.readline()
        print("Exiting...")
        return True

    sel = selectors.DefaultSelector()
    sel.register(camstate.cm.event_fd, selectors.EVENT_READ, camstate.handle_req)
    sel.register(kmsstate.card.fd, selectors.EVENT_READ, kmsstate.readdrm)
    sel.register(sys.stdin, selectors.EVENT_READ, handle_key_event)

    running = True

    while running:
        events = sel.select()
        for key, mask in events:
            # If the handler return True, we should exit
            if key.data():
                print("exit")
                running = False

    camstate.cam.stop()
    camstate.cam.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
