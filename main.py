import cv2 as cv
import numpy as np
import math

cap = cv.VideoCapture(0)
     
while (1):
    try:
        ret, frame = cap.read()
        frame=cv.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)

        roi=frame[100:300, 100:300]

        cv.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

        lower_skin = np.array([0, 20, 70], dtype = np.uint8)
        upper_skin = np.array([20, 255, 255], dtype = np.uint8)

        mask = cv.inRange(hsv, lower_skin, upper_skin)
        mask = cv.dilate(mask,kernel,iterations = 4)
        mask = cv.GaussianBlur(mask,(5,5),100) 

        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        cnt = max(contours, key = lambda x: cv.contourArea(x))

        epsilon = 0.0005 * cv.arcLength(cnt, True)
        approx= cv.approxPolyDP(cnt,epsilon, True)

        hull = cv.convexHull(cnt)
        areahull = cv.contourArea(hull)
        areacnt = cv.contourArea(cnt)

        arearatio = ((areahull-areacnt) / areacnt) * 100

        hull = cv.convexHull(approx, returnPoints = False)
        defects = cv.convexityDefects(approx, hull)

        l = 0
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            d = (2 * ar) / a

            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            if angle <= 90 and d > 30:
                l += 1
                cv.circle(roi, far, 3, [255, 0, 0], -1)
            cv.line(roi, start, end, [0, 255, 0], 2)

        l += 1

        font = cv.FONT_HERSHEY_SIMPLEX

        if l == 1:
            if areacnt < 2000:
                cv.putText(frame,'Esperando', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
            else:
                executado = False
                if arearatio < 12 and not executado:
                    cv.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
                    executado = True
                elif arearatio < 17.5:
                    cv.putText(frame, '?', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
                else:
                    cv.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
        elif l == 2:
            cv.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
        elif l == 3:
              if arearatio < 27:
                    cv.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
              else:
                    cv.putText(frame, 'OK!', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)   
        elif l == 4:
            cv.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
        elif l == 5:
            cv.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)
        else:
            cv.putText(frame, 'Reposione o Objeto', (10, 50), font, 2, (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow('Preto e Branco', mask)
        cv.imshow('OpenCV - Webcam', frame)

    except:
        pass
    k = cv.waitKey(5) & 0xFF

    if k == 27:
        break

cv.destroyAllWindows()
cap.release()
