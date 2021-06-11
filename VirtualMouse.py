import cv2 as cv
import mediapipe as mp
import time
import HandRecognitionModule as hrm
import autopy
import numpy as np

windowX, windowY = 640, 480
screenX, screenY = autopy.screen.size()
rectangleX, rectangleY = windowX - 100, windowY - 100

smootheningFactor = 7

plocX, plocY = 0, 0
clocX, clocY = 0, 0

capture = cv.VideoCapture(0)
capture.set(3, windowX)
capture.set(4, windowY)

detector = hrm.handDetector(decetion_conf=0.75, tracking_conf=0.65)

pTime = 0
while True:
    isTrue, img = capture.read()
    img = detector.findHands(img)
    lmList = detector.getPosition(img)
    cv.rectangle(img, (100, 100), (rectangleX, rectangleY), (0, 0, 0))

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[4][1:]

        fingersUpList = detector.fingerUp(lmList)
        if (fingersUpList[1] == 1) and (fingersUpList[2] == 0):

            x3 = np.interp(x1, (100, windowX - 100), (0, screenX))
            y3 = np.interp(y1, (100, windowY - 100), (0, screenY))

            clocX = plocX + (x3 - plocX) / smootheningFactor
            clocY = plocY + (y3 - plocY) / smootheningFactor
            plocY = clocY
            plocX = clocX

            autopy.mouse.move(screenX - clocX, clocY)

        if (fingersUpList[1] == 1) and (fingersUpList[2] == 1):
            length = detector.getDistance(img, lmList, 12, 8)
            if length < 40:
                autopy.mouse.click()

    # setting FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'FPS:{int(fps)}', (20, 50), cv.FONT_ITALIC, 1, (0, 0 ,0), 2)

    cv.imshow('WebCam', img)
    cv.waitKey(1)