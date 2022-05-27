from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle

import io
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from starlette.middleware.cors import CORSMiddleware
from datetime import datetime

now = datetime.now()

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Load face names and encoding from pickle
classNames, known_face_encodings = pickle.load(open('faces.p', 'rb'))

@app.post("/faces_recognition/")
async def faces_recognition(image_upload: UploadFile = File(...)):
    data = await image_upload.read()

    # Load an image
    image = Image.open(io.BytesIO(data))

    # Detect face(s) and encode them
    face_locations = face_recognition.face_locations(np.array(image))
    face_encodings = face_recognition.face_encodings(np.array(image), face_locations)

    face_names = []
    draw = ImageDraw.Draw(image)

    # Recognize face(s)
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = classNames[best_match_index]
            top, right, bottom, left = face_location
            draw.rectangle([left, top, right, bottom])
            draw.text((left, top), name)
            face_names.append(name)
        else:
            top, right, bottom, left = face_location
            draw.rectangle([left, top, right, bottom])
            draw.text((left, top), "Unknown")
            face_names.append("Unknown"+str(now.strftime("%m/%d/%Y, %H:%M:%S")))

    return {"faces": face_names}