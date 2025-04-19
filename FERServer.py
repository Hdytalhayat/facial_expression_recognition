import sys
import argparse
import copy
import datetime
import numpy as np
import cv2 as cv
import socket
from collections import Counter, deque
import threading
import time
from cvzone.HandTrackingModule import HandDetector


# Communication setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 1209)

# Ekspresi
expression = ""

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from facial_fer_model import FacialExpressionRecog

sys.path.append('face_detection_yunet/')
from yunet import YuNet

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('--model', '-m', type=str, default='./facial_expression_recognition_mobilefacenet_2022july.onnx',
                    help='Path to the facial expression recognition model.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--vis', '-v', action='store_true', help='Specify to open a window for result visualization.')
args = parser.parse_args()

# Queue to store expressions with timestamps
expression_queue = deque()

# Function to calculate the most frequent expression in the last 5 seconds
def get_average_expression():
    current_time = time.time()
    
    # Remove old expressions (older than 5 seconds)
    while expression_queue and (current_time - expression_queue[0][1]) > 5:
        expression_queue.popleft()
    
    # Get only the expressions
    expressions = [exp[0] for exp in expression_queue]
    
    if expressions:
        # Find the most common expression
        most_common_expression = Counter(expressions).most_common(1)[0][0]
        return most_common_expression
    return "No expression detected"

def visualize(image, det_res, fer_res, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    global expression_queue
    
    print('%s 1 face detected.' % datetime.datetime.now())

    output = image.copy()
    landmark_color = [
        (255,  0,   0),  # right eye
        (0,    0, 255),  # left eye
        (0,  255,   0),  # nose tip
        (255,  0, 255),  # right mouth corner
        (0,  255, 255)   # left mouth corner
    ]

    current_time = time.time()

    # Only process the first face
    if len(det_res) > 0:
        det = det_res[0]
        fer_type = fer_res[0]
        bbox = det[0:4].astype(np.int32)
        fer_type = FacialExpressionRecog.getDesc(fer_type)
        print("Face: %d %d %d %d %s." % (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], fer_type))
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        cv.putText(output, fer_type, (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        landmarks = det[4:14].astype(np.int32).reshape((5, 2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)
        
        # Add detected expression to the queue with timestamp
        expression_queue.append((fer_type, current_time))
    
    # Calculate the most common expression in the last 5 seconds
    average_expression = get_average_expression()
    print("Average Expression (5 sec):", average_expression)
    
    # Send result through socket
    if len(expression_queue) > 0:
        sock.sendto(str.encode(str(average_expression)), serverAddressPort)
    
    return output

def process(detect_model, fer_model, frame):
    h, w, _ = frame.shape
    detect_model.setInputSize([w, h])
    dets = detect_model.infer(frame)

    if dets is None or len(dets) == 0:
        return False, None, None

    # Only process the first face
    dets = dets[:1]
    fer_res = fer_model.infer(frame, dets[0][:-1])
    return True, dets, fer_res

if __name__ == '__main__':
    try:
        backend_id = backend_target_pairs[args.backend_target][0]
        target_id = backend_target_pairs[args.backend_target][1]

        detect_model = YuNet(modelPath='face_detection_yunet/face_detection_yunet_2023mar.onnx')
        fer_model = FacialExpressionRecog(modelPath=args.model, backendId=backend_id, targetId=target_id)
        pTime = 0
        plocX, plocY = 0, 0
        clocX, clocY = 0, 0
        smoothening = 4
        # Use default camera as input
        deviceId = 0
        cap = cv.VideoCapture(deviceId)
        cap.set(cv.CAP_PROP_FPS, 24)
        camW, camH = 640, 480
        frameR = 100
        detector = HandDetector(detectionCon=0.65, maxHands=1)
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            frame = cv.flip(frame, 1)
            
            hands, frame = detector.findHands(frame)
            
            cv.rectangle(frame,(frameR,frameR), (camW - frameR, camH - frameR), (255,0,255),2)
            
            # if hands:
            #     lmlist = hands[0]['lmList']
            #     ind_x, ind_y = lmlist[8][0], lmlist[8][1]
            #     mid_x, mid_y = lmlist[12][0], lmlist[12][1] 
            #     cv.circle(frame, (ind_x,ind_y), 5,(0,255,255),2)
            #     fingers = detector.fingersUp(hands[0])
            #     if fingers[1] == 1 and fingers[2] == 0:
            #         conv_x = int(np.interp(ind_x,(frameR, camW-frameR),(0,1920)))
            #         conv_y = int(np.interp(ind_y,(frameR, camH-frameR),(0,1080)))
            #         clocX = plocX + (conv_x - plocX) / smoothening
            #         clocY = plocY + (conv_y - plocY) / smoothening
            #         mouse.move(clocX,clocY)
            #         mouse.release(button='left')
            #         plocX, plocY = clocX, clocY
            #     if fingers[1] == 1 and fingers[2] == 1:
            #         conv_x = int(np.interp(ind_x,(frameR, camW-frameR),(0,1920)))
            #         conv_y = int(np.interp(ind_y,(frameR, camH-frameR),(0,1080)))
            #         clocX = plocX + (conv_x - plocX) / smoothening
            #         clocY = plocY + (conv_y - plocY) / smoothening
            #         mouse.move(clocX,clocY)
            #         plocX, plocY = clocX, clocY
            #         if abs(ind_x-mid_x)<25:
            #             mouse.press(button='left')
            if not hasFrame:
                print('No frames grabbed!')
                break

            # Get detection and fer results
            status, dets, fer_res = process(detect_model, fer_model, frame)

            if status:
                # Draw results on the input image
                frame = visualize(frame, dets, fer_res)

            # Visualize results in a new window
            if args.vis:
                cv.imshow('FER Demo', frame)

        cap.release()
        cv.destroyAllWindows()

    except Exception as e:
        print(f"Error occurred: {e}")