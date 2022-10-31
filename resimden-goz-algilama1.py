import cv2

img = cv2.imread("eye.png")

face_cascade = cv2.CascadeClassifier("frontalface.xml")
eye_cascade = cv2.CascadeClassifier("eye.xml")

