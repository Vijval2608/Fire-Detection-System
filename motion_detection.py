import cv2
import numpy as np

cap = cv2.VideoCapture("fire_video3.mp4")

fgbg = cv2.createBackgroundSubtractorMOG2()

target_size = (640, 480)

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, target_size)

    fgmask = fgbg.apply(frame)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            print("Fire detected!")

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
