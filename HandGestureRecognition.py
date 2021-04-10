import cv2
import time
import os
import numpy as np
import pyautogui

import hand_recognition as htm

W, H = 640, 480

folder = "source"

cap = cv2.VideoCapture(0)
cap.set(3, W)
cap.set(4, H)

listImage = os.listdir(folder)
cv2Image = [cv2.imread(f'{folder}/{image}') for image in listImage]
k = 1

cTime = 0
pTime = 0

index = [4, 8, 12, 16, 20]
detector = htm.HandDetector()

typeHands = dict([(0, 'down'), (1,'left'), (2, 'right'), (3, 'Trois'), (4,'Quatre'), (5, 'Cinq'), (6, 'Ok'), (7, 'Like')])
print(typeHands)

def key_down(given_key):
    keys = ["left", "right", "up", "down"]
    for key in keys:
        if given_key != key:
            pyautogui.keyUp(key)
        else:
            pyautogui.keyDown(given_key)

for img in cv2Image:
    output = cv2.resize(img, (200, 200))
    cv2.imwrite(f"image/{k}.jpg", output)
    k+=1

listImage = os.listdir("image")
cv2Image = [cv2.imread(f'image/{img}') for img in listImage]
detector = htm.HandDetector()
print(len(cv2Image))

detector = htm.HandDetector()
while True:

    success, img = cap.read()
    if(success):
        w, h , c = cv2Image[0].shape
        number_letter = ""

        img, skeleton = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            fingerUpIndex = []
            
            # THUMB_IP
            if(lmList[index[0]][1] > lmList[index[0] - 1][1]):
                fingerUpIndex.append(1)
            else:
                fingerUpIndex.append(0)

            # Other fingers
            for i in range(len(index)):
                if(lmList[index[i]][2] < lmList[index[i] - 2][2]):
                    fingerUpIndex.append(1)
                else:
                    fingerUpIndex.append(0)

            number = len([val for val in fingerUpIndex if val != 0])
            if len(fingerUpIndex) != 0:
                indices = [i for i, value in enumerate(fingerUpIndex) if value == 1]
            if(number == 1):
                key_down('left')
            elif(number == 2):
                key_down('right')
            elif(number == 5):
               key_down('down')
            if indices == [0,1]:
                key_down('up')
        # img[0:h, 0:w] = cv2Image[0]
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        # print(fingerUpIndex.index(1))
        # print(number_letter)
        # tmp = np.ones((640,480,3), dtype=np.uint8)
        cv2.putText(img, str(int(fps)), (480, 60),  3, cv2.FONT_HERSHEY_PLAIN, (255, 0, 255))
        cv2.putText(img, number_letter, (320, 60),  3, cv2.FONT_HERSHEY_PLAIN, (255, 200, 180))
        cv2.imshow("image", img)
        cv2.imshow("screen_2", skeleton)
        # cv2.imshow("screen_3", tmp)
        cv2.waitKey(1)