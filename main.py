import cv2 
import threading
import numpy as np
import pygame
import os

classes = ['Weapon']
model_weights = 'weapon.weights'
model_cfg = 'weapons.cfg'
model = cv2.dnn.readNet(model_weights, model_cfg)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
cam = cv2.VideoCapture(0) 
lock = threading.Lock()
is_processing = False
match = False
pygame.mixer.init()
sound = 'alarm.wav'
boxes = []
confidences = []
class_ids = []
output = ()
shared_boxes = []
shared_labels = []
shared_lock = threading.Lock()



def cam_runner():
    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            raise Exception("Error reading frame")

        if count % 30 == 0:
            threading.Thread(target=process_img, args=(frame,)).start()

        count += 1
        with shared_lock:
            for box, label in zip(shared_boxes, shared_labels):
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Weapon Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def process_img(frame): 
    global output
    blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    with lock:
        output = model.forward(model.getUnconnectedOutLayersNames())

    threading.Thread(target=process_detection, args=(frame, output)).start()

def process_detection(frame, output):
    global boxes, confidences, class_ids, match
    boxes.clear()
    confidences.clear()
    class_ids.clear()
    height, width = frame.shape[:2]
    for result in output:
        for detection in result:
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                if not os.path.exists('weapon.jpg'):
                    cv2.imwrite('weapon.jpg', frame)
                match = True
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                play_sound()

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    with shared_lock:
        shared_boxes.clear()
        shared_labels.clear()
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                label = classes[class_ids[i]]
                shared_boxes.append(box)
                shared_labels.append(label)

def process_saved_img():
    img = cv2.imread(input('Enter image path: '))   
    process_img(img)
    

def play_sound():
    global is_processing 
    if not is_processing:
        threading.Thread(target=ring).start()


def ring():
    global is_processing
    is_processing = True
    pygame.mixer.music.load(sound)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    is_processing = False

if __name__ == '__main__':
    cmd = int(input('Enter 1 to check live camera and 2 to load saved image: '))
    if cmd == 1:
        cam_runner()
    elif cmd == 2:
        process_saved_img()
    else:
        print('Invalid input')