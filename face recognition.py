import os


import cv2
import numpy as np
import face_recognition
from datetime import datetime
import csv

known_face_encoding = []
known_face_name = []
video_capture = cv2.VideoCapture(0)
address = "\Python\Face Recognition\Image"
a = os.listdir(address)
for image in a:
    img = face_recognition.load_image_file("Image/"+image)
    encoding = face_recognition.face_encodings(img)[0]
    known_face_encoding.append(encoding)
    known_face_name.append(os.path.basename(image).split(".")[0])

print(known_face_name)
student = known_face_name.copy()

face_location = []
face_encodings = []
face_name = []
s = True

now = datetime.now()
exact = now.strftime("%Y-%m-%d")

f = open(exact + ".csv", "w+", newline="")
nwrite = csv.writer(f)

while True:
    _, frame = video_capture.read()
    resize_image = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    if s:
        face_location = face_recognition.face_locations(rgb_resize_image)
        face_encodings = face_recognition.face_encodings(rgb_resize_image, face_location)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_face_match = np.argmin(face_distance)
            if matches[best_face_match]:
                name = known_face_name[best_face_match]
            face_name.append(name)
            if name in known_face_name:
                if name in student:
                    student.remove(name)
                    print(student)
                    exact_time = now.strftime("%H:%M:%S")
                    nwrite.writerow([name, exact_time])
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()
