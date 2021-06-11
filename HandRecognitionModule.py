import cv2 as cv
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self,mode = False, max_hands = 2, decetion_conf = 0.5, tracking_conf = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.decetion_conf = decetion_conf
        self.tracking_conf = tracking_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.decetion_conf, self.tracking_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def getPosition(self, img, handNo = 0, draw = False):
        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(w * lm.x), int(h * lm.y)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx,cy), 6, (255,255,255), -1)
        return lmList

    def fingerUp(self, lmList):
        finger_up_list = [0] * 5
        fingers = [8, 12, 16, 20]
        for i,finger in enumerate(fingers):
            if lmList[finger][2] < lmList[finger - 1][2]:
                finger_up_list[i + 1] = 1
                finger_up_list[0] = 1
            else:
                finger_up_list[i + 1] = 0
        return finger_up_list

    def getDistance(self, img,lmList, index1, index2, draw = True):
        finger1, finger2= lmList[index1], lmList[index2]
        x1, y1 = lmList[index1][1], lmList[index1][2]
        x2, y2 = lmList[index2][1], lmList[index2][2]

        length = math.hypot(x2 - x1, y2 - y1)

        if draw:
            cx , cy = (x2 + x1) // 2, (y2 + y1) // 2
            cv.circle(img, (x1, y1), 10, (255, 0, 255), -1)
            cv.circle(img, (x2, y2), 10, (255, 0, 255), -1)
            cv.circle(img, (cx, cy), 10, (255, 0, 255), -1)
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            if length <= 40:
                cv.circle(img, (cx, cy), 10, (0, 255, 0), -1)

        return length

def main():
    capture = cv.VideoCapture(0)
    pTime = 0
    cTime = 0

    detector = handDetector(decetion_conf=0.75, tracking_conf=0.65)
    while True:
        isTrue, img = capture.read()
        img = detector.findHands(img)
        lmList = detector.getPosition(img, draw = False)

        if len(lmList) != 0:
            finger_up_list = detector.fingerUp(lmList)
            # print(finger_up_list)
            length = detector.getDistance(img, lmList, 8, 12)
            # print(length)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_ITALIC, 2, (0, 0, 0), 3)

        cv.imshow('WebCam', img)
        cv.waitKey(1)


if __name__ == '__main__':
    main()