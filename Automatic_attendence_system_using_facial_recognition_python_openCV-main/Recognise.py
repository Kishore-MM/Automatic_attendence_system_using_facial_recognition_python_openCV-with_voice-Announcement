import cv2
import numpy as np
from datetime import datetime
import xlwt
from xlrd import open_workbook
from xlutils.copy import copy
import os
import pyttsx3

# Function to assure path existence
def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to initialize the text-to-speech engine
def init_tts_engine():
    engine = pyttsx3.init()
    return engine

# Function to speak a given text
def speak(engine, text):
    engine.say(text)
    engine.runAndWait()

# Function to recognize faces and write attendance
def recognize_and_write():
    cap = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    attendance_dict = {}
    font = cv2.FONT_HERSHEY_SIMPLEX
    engine = init_tts_engine()

    while True:
        ret, img = cap.read()

        if not ret or img is None:
            print("Error: Failed to capture image from camera.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 7)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id, conf = recognizer.predict(roi_gray)

            if conf < 50:
                # Fetch name from face_data based on id
                name = get_name_from_id(str(id))
                if name:
                    cv2.putText(img, f"Welcome to class, {name}", (x, y - 10), font, 0.55, (120, 255, 120), 1)
                    if str(id) not in attendance_dict:
                        filename = output('attendance', 'class1', id, name, 'yes')
                        attendance_dict[str(id)] = name
                        speak(engine, f"Welcome to class, {name}")
                else:
                    name = 'Unknown'
                    cv2.putText(img, f"You're not recognized. Please visit our college admission.", (x, y - 10), font, 0.55, (120, 255, 120), 1)
                    speak(engine, "You're not registered.")
            else:
                name = 'Unknown'
                cv2.putText(img, f"You're not recognized. Please visit our college admission.", (x, y - 10), font, 0.55, (120, 255, 120), 1)
                speak(engine, "You're not registered.")

        cv2.imshow('frame', img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_name_from_id(id):
    # Load face data from face_data.npy
    data_path = 'face_data.npy'
    if os.path.exists(data_path):
        face_data = np.load(data_path, allow_pickle=True)
        for data in face_data:
            if data['id'] == id:
                return data['name']
    return None

def output(filename, sheet, num, name, present):
    directory = 'firebase/attendance_files/'
    assure_path_exists(directory)
    
    file_path = os.path.join(directory, f"{filename}_{str(datetime.now().date())}.xls")

    if os.path.isfile(file_path):
        rb = open_workbook(file_path)
        book = copy(rb)
        sh = book.get_sheet(0)
    else:
        book = xlwt.Workbook()
        sh = book.add_sheet(sheet)
        sh.write(0, 0, datetime.now().date(), xlwt.easyxf(num_format_str='D-MMM-YY'))
        sh.write(1, 0, 'Name', xlwt.easyxf('font: name Times New Roman, color-index red, bold on'))
        sh.write(1, 1, 'Present', xlwt.easyxf('font: name Times New Roman, color-index red, bold on'))
        sh.write(1, 2, 'Time', xlwt.easyxf('font: name Times New Roman, color-index red, bold on'))

    current_time = datetime.now().strftime("%H:%M:%S")
    sh.write(num + 1, 0, name)
    sh.write(num + 1, 1, present)
    sh.write(num + 1, 2, current_time)

    book.save(file_path)
    return file_path

if __name__ == "__main__":
    recognize_and_write()
