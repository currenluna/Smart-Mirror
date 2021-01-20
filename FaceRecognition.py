import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(1)

while True:
    ret, frame = video_capture.read()
    print(ret)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    img_flip = cv2.flip(frame, 1)

    # Draws a black background
    height, width = frame.shape[:2]
    # cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 0), -1)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_flip, (x, y), (x+w, y+h), (255, 255, 255), 10)

    cv2.imshow('Video', img_flip)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# import time
# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture(1)
# time.sleep(1)
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
