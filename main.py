from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from enum import Enum
from database import get_connection, get_workers, save_attendance, save_face_embedding, get_all_embeddings, init_pool, release_connection
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from typing import List
from typing import Optional
from datetime import datetime
import faiss

app = FastAPI()

# --- Daha hafif model, daha küçük det_size ---
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))

# --- Global embedding cache ---
"""embedding_cache: list = []"""
embedding_cache = []
faiss_index = None
id_map = []


"""def load_embedding_cache():
    global embedding_cache
    kayitlar = get_all_embeddings()
    for kayit in kayitlar:
        kayit["embedding"] = np.array(kayit["embedding"], dtype=np.float32)
    embedding_cache = kayitlar
    print(f"[Cache] {len(embedding_cache)} kişi yüklendi.")"""

def load_embedding_cache():
    global embedding_cache, faiss_index, id_map

    kayitlar = get_all_embeddings()

    embeddings = []
    id_map = []

    for kayit in kayitlar:
        emb = np.array(kayit["embedding"], dtype=np.float32)
        embeddings.append(emb)
        id_map.append(kayit)

    if len(embeddings) == 0:
        return

    embeddings = np.array(embeddings).astype("float32")

    # Normalize (cosine için önemli)
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(d)  # cosine similarity

    faiss_index.add(embeddings)

    embedding_cache = kayitlar

    print(f"[FAISS] {len(embedding_cache)} kişi yüklendi.")







@app.on_event("startup")
async def startup_event():
    init_pool()
    load_embedding_cache()


# ─── Enum'lar ────────────────────────────────────────────────────────────────

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
    custom_time: Optional[datetime] = None  # test için


# ─── Yardımcı fonksiyon ───────────────────────────────────────────────────────

def decode_image(contents: bytes):
    """Byte → OpenCV görüntüsü. Büyük görüntüleri 640px genişliğe küçültür."""
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > 640:
        scale = 640 / w
        img = cv2.resize(img, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# ─── Endpoint'ler ─────────────────────────────────────────────────────────────




@app.get("/last-action/{user_id}")
def son_kayit(user_id: int):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT "Type", "Time" FROM "Attendances" '
            'WHERE "UserId" = %s '
            'ORDER BY "Time" DESC LIMIT 1',
            (user_id,)
        )
        row = cursor.fetchone()
        cursor.close()
    finally:
        release_connection(conn)

    if row is None:
        return {"nextAction": "giris"}

    last_type = row[0]
    last_time = row[1]

    # Türkiye saatine çevir
    from datetime import timezone, timedelta
    turkey = timezone(timedelta(hours=3))
    last_time_tr = last_time.astimezone(turkey).strftime("%H:%M")

    next_action = "cikis" if last_type == "giris" else "giris"

    return {
        "nextAction": next_action,
        "lastTime": last_time_tr,
        "lastType": last_type
    }






@app.post("/enroll/{user_id}")
async def yuz_kaydet(user_id: int, photos: List[UploadFile]):
    embeddings = []

    for photo in photos:
        contents = await photo.read()
        img = decode_image(contents)
        if img is None:
            continue

        faces = face_app.get(img)
        if len(faces) > 0:
            embeddings.append(faces[0].embedding)

    if len(embeddings) < 3:
        return {"hata": "En az 3 geçerli yüz fotoğrafı gerekli"}

    avg_embedding = np.mean(embeddings, axis=0)
    save_face_embedding(user_id, avg_embedding)
    load_embedding_cache()  # Cache'i güncelle

    return {
        "mesaj": "Yüz kaydedildi",
        "kullanilan_foto": len(embeddings)
    }


@app.get("/workers")
def isci_listesi():
    return {"workers": get_workers()}


"""@app.post("/attendance")
def kayit_ekle(veri: AttendanceRequest):
    save_attendance(
        user_id=veri.worker_id,
        event_type=veri.event_type,
        shift=veri.shift,
        description=veri.description,
        custom_time=veri.custom_time
    )
    return {"mesaj": "Kayıt veritabanına kaydedildi."}"""

@app.post("/attendance")
def kayit_ekle(veri: AttendanceRequest):
    if veri.event_type == "cikis":
        # Son girişi bul ve vardiyayı hesapla
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT "Id", "Time" FROM "Attendances" '
                'WHERE "UserId" = %s AND "Type" = \'giris\' '
                'ORDER BY "Time" DESC LIMIT 1',
                (veri.worker_id,)
            )
            row = cursor.fetchone()
            cursor.close()
        finally:
            release_connection(conn)

        if row:
            giris_id = row[0]
            giris_time = row[1]
            cikis_time = veri.custom_time if veri.custom_time else datetime.utcnow()
            
            # Vardiyayı hesapla
            hesaplanan_shift = vardiay_hesapla(giris_time, cikis_time)
            
            # Giriş satırının shift'ini güncelle
            conn = get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE "Attendances" SET "Shift" = %s WHERE "Id" = %s',
                    (hesaplanan_shift, giris_id)
                )
                conn.commit()
                cursor.close()
            finally:
                release_connection(conn)
            
            # Çıkışı kaydet
            save_attendance(
                user_id=veri.worker_id,
                event_type=veri.event_type,
                shift=hesaplanan_shift,
                description=veri.description,
                custom_time=veri.custom_time
            )
        else:
            # Eşleşen giriş yoksa direkt kaydet
            save_attendance(
                user_id=veri.worker_id,
                event_type=veri.event_type,
                shift=veri.shift,
                description=veri.description,
                custom_time=veri.custom_time
            )
    else:
        # Giriş kaydı — shift geçici olarak Kotlin'den geleni kullan
        save_attendance(
            user_id=veri.worker_id,
            event_type=veri.event_type,
            shift=veri.shift,
            description=veri.description,
            custom_time=veri.custom_time
        )
    
    return {"mesaj": "Kayıt veritabanına kaydedildi."}




def vardiay_hesapla(giris_time: datetime, cikis_time: datetime) -> int:
    # UTC'den Türkiye saatine çevir
    from datetime import timezone, timedelta
    turkey = timezone(timedelta(hours=3))
    
    giris_tr = giris_time.astimezone(turkey)
    cikis_tr = cikis_time.astimezone(turkey)
    
    # Vardiya dilimleri (saat olarak)
    # Sabah: 08:00-16:00 (shift=2)
    # Akşam: 16:00-00:00 (shift=3)
    # Gece:  00:00-08:00 (shift=1)
    
    calisma_suresi = (cikis_tr - giris_tr).total_seconds() / 3600
    
    # Her vardiyayla örtüşen saati hesapla
    def ortusme_hesapla(giris, cikis, vardiya_baslangic, vardiya_bitis):
        # Günü normalize et
        from datetime import date
        gun = giris.date()
        
        vb = giris.replace(
            hour=vardiya_baslangic, minute=0, second=0, microsecond=0
        )
        vbt = giris.replace(
            hour=vardiya_bitis, minute=0, second=0, microsecond=0
        )
        
        # Gece vardiyası gece yarısını geçiyor
        if vardiya_bitis <= vardiya_baslangic:
            vbt = vbt + timedelta(days=1)
        
        # Örtüşme hesapla
        baslangic = max(giris, vb)
        bitis = min(cikis, vbt)
        
        if bitis > baslangic:
            return (bitis - baslangic).total_seconds() / 3600
        return 0
    
    sabah = ortusme_hesapla(giris_tr, cikis_tr, 8, 16)   # 08:00-16:00
    aksam = ortusme_hesapla(giris_tr, cikis_tr, 16, 24)  # 16:00-00:00
    gece  = ortusme_hesapla(giris_tr, cikis_tr, 0, 8)    # 00:00-08:00
    
    # En fazla örtüşen vardiyayı seç
    maksimum = max(sabah, aksam, gece)
    
    if maksimum == sabah:
        return 2  # Sabah
    elif maksimum == aksam:
        return 3  # Akşam
    else:
        return 1  # Gece



@app.get("/workers/embeddings")
def isci_embedding_listesi():
    kayitlar = get_all_embeddings()
    return {"workers": kayitlar}


@app.post("/embed")
async def embedding_cikar(photo: UploadFile):
    contents = await photo.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    faces = face_app.get(img)
    if len(faces) == 0:
        return {"found": False}
    
    embedding = faces[0].embedding
    # Normalize et
    embedding = embedding / np.linalg.norm(embedding)
    return {"found": True, "embedding": embedding.tolist()}



@app.post("/recognize")
async def yuz_tani(photo: UploadFile):
    global faiss_index, id_map

    contents = await photo.read()
    img = decode_image(contents)

    if img is None:
        return {"tanindi": False, "mesaj": "Görüntü okunamadı"}

    faces = face_app.get(img)
    if len(faces) == 0:
        return {"tanindi": False, "mesaj": "Yüz tespit edilemedi"}

    embedding = faces[0].embedding.astype(np.float32)

    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    embedding = np.expand_dims(embedding, axis=0)

    if faiss_index is None:
        return {"tanindi": False, "mesaj": "Sistem hazır değil"}

    D, I = faiss_index.search(embedding, k=1)

    skor = float(D[0][0])
    index = int(I[0][0])

    if skor > 0.5:
        kisi = id_map[index]
        return {
            "tanindi": True,
            "id": kisi["id"],
            "full_name": kisi["full_name"],
            "skor": round(skor, 3)
        }

    return {"tanindi": False, "mesaj": "Tanınamadı"}


@app.delete("/enroll/{user_id}")
def yuz_sil(user_id: int):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE "Users" SET "FaceEmbedding" = NULL WHERE "Id" = %s',
            (user_id,)
        )
        conn.commit()
        cursor.close()
    finally:
        release_connection(conn)

    load_embedding_cache()  # Cache'i güncelle
    return {"mesaj": "Yüz silindi"}


@app.post("/reload-cache")
def cache_yenile():
    """Cache'i manuel yenilemek için (isteğe bağlı)."""
    load_embedding_cache()
    return {"mesaj": f"Cache yenilendi, {len(embedding_cache)} kişi yüklendi."}
