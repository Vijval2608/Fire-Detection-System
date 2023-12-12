import cv2
import numpy as np

video = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = video.read()

    if not ret:
        break

    frame = cv2.resize(frame, (500, 500))
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower_flame = np.array([5, 44, 228])
    upper_flame = np.array([25, 255, 255])

    flame_mask = cv2.inRange(hsv, lower_flame, upper_flame)

    fgmask = fgbg.apply(frame)

    combined_mask = cv2.bitwise_and(fgmask, flame_mask)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    flames_only = cv2.bitwise_and(frame, frame, mask=flame_mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            x, y, w, h = cv2.boundingRect(contour)

            # Draw the rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (96, 225, 0), 2)
            cv2.putText(frame, 'FIRE', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (96,225,0), 2)
            

    cv2.imshow("Flame Motion Detection", frame)
    cv2.imshow("Flames Only", flames_only)

    if cv2.waitKey(20) == 27:
        break

video.release()
cv2.destroyAllWindows()
