"""
Hold mouse side button 1 to automatically aim at the target closest to the center of the screen
"""
import math
import kmNet  # Import kmNet module
from myReadVideoThread import VideoCapture # Read capture card data in a threaded way to ensure the inference image is always up-to-date and avoid cumulative delay
import sys
from pathlib import Path
import cv2
import torch
import time
video_num = 0  # Capture card number: 0, 1, 2...

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
from models.experimental import attempt_load
from utils.general import is_ascii, non_max_suppression, scale_boxes, set_logging
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


'''
Convert deflection angles to mouse coordinates
'''
def fov(ax, ay, px, py, Cx, Cy):
    dx = (ax - px / 2) * 3
    dy = (ay - py / 2) * 2.25
    Rx = Cx / 2 / math.pi
    Ry = Cy / 2 / math.pi
    mx = math.atan2(dx, Rx) * Rx
    my = (math.atan2(dy, math.sqrt(dx * dx + Rx * Rx))) * Ry
    return mx, my

def show_result_to_lcd(x, y, ex, ey, frame):
#  Center display the aimed target
    pic = cv2.resize(frame, (128, 160))  # Full-screen display requires resolution of 128x160
    pic = pic.flatten()     # Format
    kmNet.lcd_picture(pic)  # Display on the box screen

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        line_thickness=1,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        ):
    # Initialize
    set_logging()
    device = select_device(device)
    print(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, device=device)  # load FP32 model
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)
    cap = VideoCapture(video_num)  # Start capturing images

    global xyxy        # Single target
    global time_aim    # Last aiming time
    aim_process = 0
    while True:
        # Get one frame
        timestamp_t0 = time.time()
        frame = cap.read()  # Get the latest frame image
        img = torch.from_numpy(frame).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        img = img.transpose(2, 3)
        img = img.transpose(1, 2)
        timestamp_t1 = time.time()
        # Inference
        pred = model(img, augment=False, visualize=False)[0]
        timestamp_t2 = time.time()
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        timestamp_t3 = time.time()

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            s = ''
            result = ''
            annotator = Annotator(frame, line_width=line_thickness, pil=not ascii)
            if len(det):    # If the number of prediction boxes is greater than 0
                # Convert the coordinates of the prediction box from normalized coordinates to original coordinates, im.shape[2:] is the width and height of the image, det[:, :4] is the coordinates of the prediction box, frame.shape is the width and height of the image
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += str(n.item()) + ' ' + str(names[int(c)]) + ' '  # add to string
                # print('CLASS:' + s)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # print("\t" + names[int(c)], "(", int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), ")")
        timestamp_t4 = time.time()

        if kmNet.isdown_side1() == 1 and aim_process == 0:  # If the side button is pressed, it means aiming is needed at this moment. aim_process controls the aiming interval.
            '''
            Here, add coordinate conversion algorithms, mouse movement algorithms, filtering algorithms, tracking algorithms, etc.
            For simplicity, this demo assumes that there is only one target on the screen. Control the mouse to move to 80% of the target height (head). If there are multiple targets, please filter them yourself.
            '''
            # step1: Calculate the scale. The input image resolution is 1920x1080, and the capture output resolution is 640x480
            sx = int(xyxy[0])  # Real physical screen x starting position
            sy = int(xyxy[1])  # Real physical screen y starting position
            ex = int(xyxy[2])  # Real physical screen x ending position
            ey = int(xyxy[3])  # Real physical screen y ending position
            # step2: With the screen center as the reticle, calculate the offset needed to move the mouse to 80% of the rectangle height
            dx = sx + (ex - sx) / 2  # Midpoint of the rectangle x
            dy = sy + (ey - sy) * 0.2  # 20% height from the top of the rectangle
            mx, my = fov(dx, dy, 640, 480, 5140, 5140)
            print("\tMouse move coordinates", mx, my)
            kmNet.move(int(mx), int(my))  # This is just a demo. Don't write it like this. Locking a frame will definitely cause abnormal keyboard and mouse data. Add your own trajectory algorithm.
            aim_process = 1
            time_aim = time.time()
            # Display the hit target ---- The following can be shielded to speed up aiming. Displaying is the most time-consuming operation.
            pic = cv2.resize(frame, (128, 160))  # Full-screen display requires a resolution of 128x160
            pic = pic.flatten()     # Format
            kmNet.lcd_picture(pic)  # Display on the box screen

        if aim_process == 1 and kmNet.isdown_side1() == 1:  # Aiming button is pressed and not released, do not switch targets
            if (time.time() - time_aim) * 1000 >= 100:
                aim_process = 0

        if aim_process == 1 and kmNet.isdown_side1() == 0:  # Aiming button is released, reset the state machine, and wait for the next aiming press
            aim_process = 0
            pass

        cv2.imshow('frame', frame)
        timestamp_t5 = time.time()
        cv2.waitKey(0)

def main():
    kmNet.init("192.168.2.188", "32770", "8147E04E")  # Connect to the box --- If it can't connect, first check if the network is smooth
    kmNet.monitor(6666)  # Turn on physical keyboard and mouse monitoring function, listen on port 6666 --- Listening port can be arbitrary, as long as it doesn't conflict with other system ports
    kmNet.mask_side1(1)  # Mask mouse side button 1 --- The side button press message will not be sent to the game machine. But AI can detect the side button press. Used as an aim assist switch
    kmNet.trace(0, 80)   # Use hardware curve correction algorithm. Complete aiming movement within 80ms
    run()  # Start capturing inference

if __name__ == "__main__":
    main()
``
