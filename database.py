import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()   # .env dosyasını okur

DATABASE_URL =os.getenv("DATABASE_URL")  # .env dosyasından şifreyi alır

def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn



def get_workers():
    conn = get_connection()
    cursor = conn.cursor()  # sorgu çalıştırır.
    cursor.execute('SELECT "Id", "FullName", "Username" FROM "Users"')
    rows = cursor.fetchall()  # tüm sonuçları al, her satır bir tuple olarak gelir
    cursor.close()
    conn.close()

    workers = []
    for row in rows:
        workers.append(
            {
                "id": row[0],
                "full_name": row[1],
                "username": row[2]
            }
        )
    return workers


def save_attendance(user_id, event_type, shift, description):
    conn = get_connection()
    cursor = conn.cursor()
 cursor.execute(
    'INSERT INTO "Attendances" ("UserId", "Type", "Shift", "Time", "IsLate", "LateReason") VALUES (%s, %s, %s, NOW() AT TIME ZONE \'UTC\' AT TIME ZONE \'Europe/Istanbul\', %s, %s)',
    (user_id, event_type, shift, False, description)
)
    conn.commit()  # verileri kaydetme
    cursor.close()
    conn.close()




def save_face_embedding(user_id, embedding):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE "Users" SET "FaceEmbedding" = %s WHERE "Id" = %s',
        (embedding.tolist(), user_id)
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_all_embeddings():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT "Id", "FullName", "FaceEmbedding" FROM "Users" WHERE "FaceEmbedding" IS NOT NULL AND "IsDeleted" = false')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    result = []
    for row in rows:
        result.append({
            "id": row[0],
            "full_name": row[1],
            "embedding": row[2]
        })

    return result
