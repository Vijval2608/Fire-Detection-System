import cv2
import numpy as np
import matplotlib.pyplot as plt

video = cv2.VideoCapture("fire_video.mp4")

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (500,500))
    blur = cv2.GaussianBlur(frame, (15,15), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [5, 44, 228]
    upper = [25, 255, 255]
    lower = np.array(lower, dtype='uint8')
    upper = np.array(upper, dtype='uint8')

    mask = cv2.inRange(hsv, lower, upper)

    output = cv2.bitwise_and(frame, hsv, mask=mask)

    if ret == False:
        break

    cv2.imshow("Output", output)

    if cv2.waitKey(20) == 27:
        break

video.release()
cv2.destroyAllWindows()