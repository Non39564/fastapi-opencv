from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle, os, cv2

path = 'images'

images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)
    
print(classNames)
print(encoded_face_train)

# Dump face names and encoding to pickle
pickle.dump((classNames, encoded_face_train), open('faces.p', 'wb'))