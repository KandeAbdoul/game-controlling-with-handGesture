import cv2
import mediapipe as mp
import numpy as np
import time
from utilities.utils import *


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConfidence, self.trackConfidence)

        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self, img, draw=True):
        skeleton = np.zeros((640,480,3), dtype=np.uint8)
        resized_matrix = []

        draw_spec_0 = self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=3, circle_radius=3)
        draw_spec_1 = self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=5)
        scaler = np.array([640, 480])
		

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                nodes = get_nodes(handLms.landmark)
                rescaled_nodes = normalize_nodes(nodes, scaler)
                matrix = make_adjacency_matrix(rescaled_nodes)
                resized_matrix = cv2.resize((matrix * 255).astype('uint8'), (640, 480))
                self.mpDraw.draw_landmarks(skeleton, handLms, self.mpHands.HAND_CONNECTIONS, draw_spec_0, draw_spec_1)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                cv2.imshow("screen_1", resized_matrix)
                cv2.waitKey(1)
        return img, skeleton
                    

    def findPosition(self, img, handNo = 0, draw= True):
    
        lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                h, w, z = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
       
        return lmList

     
def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    
    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        img, skeleton, resized_matrix = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList)
        cv2.putText(img, str(int(fps)), (0, 60), cv2.FONT_HERSHEY_PLAIN, 3, (2555,0,255))
        # cv2.imshow("screen_2", skeleton)
        
        # cv2.imshow('image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
    