import cv2

cap = cv2.VideoCapture("eye.mp4")
face_cascade = cv2.CascadeClassifier("frontalface.xml")

eye_cascade = cv2.CascadeClassifier("eye.xml")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (480,360))
    if ret is False:
        break


