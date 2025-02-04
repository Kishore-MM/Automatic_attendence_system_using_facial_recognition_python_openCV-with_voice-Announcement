import cv2
import os
import numpy as np
from PIL import Image
import csv

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture_faces(face_id, face_name):
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    count = 0
    save_path = "dataset/"

    assure_path_exists(save_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite(f"{save_path}User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
            cv2.imshow('frame', frame)

        if cv2.waitKey(100) & 0xFF == ord('q') or count >= 60:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Successfully captured {count} images for ID: {face_id}, Name: {face_name}")

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, 'uint8')
        id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = face_cascade.detectMultiScale(img_np)

        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y + h, x:x + w])
            ids.append(id)

    return face_samples, ids

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images_and_labels('dataset')
    
    if len(faces) == 0 or len(ids) == 0:
        print("Error: No faces found for training.")
        return

    recognizer.train(faces, np.array(ids))

    assure_path_exists("trainer/")
    recognizer.write('trainer/trainer.yml')
    print("Successfully trained recognizer and saved to 'trainer/trainer.yml'")

def save_face_data(face_id, face_name):
    data_path = 'face_data.npy'
    csv_path = 'face_data.csv'
    
    if os.path.exists(data_path):
        face_data = np.load(data_path, allow_pickle=True).tolist()
    else:
        face_data = []
    
    face_data.append({'id': face_id, 'name': face_name})
    np.save(data_path, face_data)
    
    # Save to CSV
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([face_id, face_name])

if __name__ == "__main__":
    while True:
        face_id = input('Enter your ID: ')
        face_name = input('Enter your name: ')
        capture_faces(face_id, face_name)
        save_face_data(face_id, face_name)
        train_recognizer()
        more = input("Do you want to add another person? (yes/no): ")
        if more.lower() != 'yes':
            break
