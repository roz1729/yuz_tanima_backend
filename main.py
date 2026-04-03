from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from enum import Enum
from database import get_connection, get_workers, save_attendance, save_face_embedding, get_all_embeddings
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from typing import List


app = FastAPI()

face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))
#face_app.prepare(ctx_id=-1, det_size=(320, 320))


class EventType(str, Enum):
    giris = "giris"
    cikis = "cikis"


class ShiftType(int, Enum):
    sabah = 2
    ogle = 3
    gece = 1


class AttendanceRequest(BaseModel):
    worker_id: int
    event_type: EventType
    shift: ShiftType
    description: str = None


@app.post("/enroll/{user_id}")
async def yuz_kaydet(user_id: int, photos: List[UploadFile]):
    embeddings = []
    # embedding = faces[0].embedding
    # embedding = embedding / np.linalg.norm(embedding)

    for photo in photos:
        contents = await photo.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        faces = face_app.get(img)
        if len(faces) > 0:
            embeddings.append(faces[0].embedding)

    if len(embeddings) < 3:
        return {"hata": "en az 3 geçerli yüz fotoğrafı gerekli"}

    avg_embedding = np.mean(embeddings, axis=0)
    save_face_embedding(user_id, avg_embedding)


    return {
        "mesaj": "yüz kaydedildi",
        "kullanilan_foto": len(embeddings)
    }


@app.get("/workers")
def isci_listesi():
    return {"workers": get_workers()}


@app.post("/attendance")
def kayit_ekle(veri: AttendanceRequest):
    save_attendance(
        user_id=veri.worker_id,
        event_type=veri.event_type,
        shift=veri.shift,
        description=veri.description
    )
    return {"mesaj": "kayıt veritabanına kaydedildi."}



@app.post("/recognize")
async def yuz_tani(photo: UploadFile):
    contents = await photo.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    if len(faces) == 0:
        return {"tanindi": False, "mesaj": "Yüz tespit edilemedi"}

    embedding = faces[0].embedding
    #embedding = faces[0].embedding
    #embedding = embedding / np.linalg.norm(embedding)

    # cached_embeddings = []
    kayitlar = get_all_embeddings()
    if len(kayitlar) == 0:
        return {"tanindi": False, "mesaj": "Kayıtlı yüz yok"}

    en_yuksek_skor = -1
    en_yakin_kisi = None

    for kayit in kayitlar:
        db_embedding = np.array(kayit["embedding"])
        skor = float(np.dot(embedding, db_embedding) /
                    (np.linalg.norm(embedding) * np.linalg.norm(db_embedding)))
        # skor = float(np.dot(embedding, db_embedding))
        if skor > en_yuksek_skor:
            en_yuksek_skor = skor
            en_yakin_kisi = kayit

    if en_yuksek_skor > 0.4:
        return {
            "tanindi": True,
            "id": en_yakin_kisi["id"],
            "full_name": en_yakin_kisi["full_name"],
            "skor": round(en_yuksek_skor, 3)
        }
    else:
        return {"tanindi": False, "mesaj": "Tanınamadı"}




@app.delete("/enroll/{user_id}")
def yuz_sil(user_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE "Users" SET "FaceEmbedding" = NULL WHERE "Id" = %s',
        (user_id,)
    )
    conn.commit()
    cursor.close()
    conn.close()
    return {"mesaj": "Yüz silindi"}

print(app.routes)



