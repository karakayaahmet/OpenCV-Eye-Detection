import cv2

cap = cv2.VideoCapture("eye.mp4")

face_cascade = cv2.CascadeClassifier("frontalface.xml")

eye_cascade = cv2.CascadeClassifier("eye.xml")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (480,360))
    if ret is False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

    roi_frame = frame[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_frame, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

