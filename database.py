import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

connection_pool = None


def init_pool():
    global connection_pool
    connection_pool = pool.SimpleConnectionPool(1, 10, DATABASE_URL)
    print("Bağlantı havuzu oluşturuldu.")


def get_connection():
    if connection_pool is None:
        init_pool()
    return connection_pool.getconn()


def release_connection(conn):
    if connection_pool is not None:
        connection_pool.putconn(conn)


def get_workers():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT "Id", "FullName", "Username" FROM "Users"')
        rows = cursor.fetchall()
        cursor.close()
    finally:
        release_connection(conn)

    workers = []
    for row in rows:
        workers.append({
            "id": row[0],
            "full_name": row[1],
            "username": row[2]
        })
    return workers


"""def save_attendance(user_id, event_type, shift, description):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO "Attendances" ("UserId", "Type", "Shift", "Time", "IsLate", "LateReason") VALUES (%s, %s, %s, NOW(), %s, %s)',
            (user_id, event_type, shift, False, description)
        )
        conn.commit()
        cursor.close()
    finally:
        release_connection(conn)


def save_attendance(user_id, event_type, shift, description, custom_time=None):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if custom_time:
            cursor.execute(
                'INSERT INTO "Attendances" ("UserId", "Type", "Shift", "Time", "IsLate", "LateReason") VALUES (%s, %s, %s, NOW(), %s, %s)',
                (user_id, event_type, shift, custom_time, False, description)
            )
        else:
            cursor.execute(
                'INSERT INTO "Attendances" ("UserId", "Type", "Shift", "Time", "IsLate", "LateReason") '
                'VALUES (%s, %s, %s, NOW(), %s, %s)',
                (user_id, event_type, shift, False, description)
            )
        conn.commit()
        cursor.close()
    finally:
        release_connection(conn)"""

def save_attendance(user_id, event_type, shift, description, custom_time=None):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        zaman = custom_time if custom_time else datetime.now()
        cursor.execute(
            'INSERT INTO "Attendances" ("UserId", "Type", "Shift", "Time", "IsLate", "LateReason") '
            'VALUES (%s, %s, %s, %s, %s, %s)',
            (user_id, event_type, shift, zaman, False, description)
        )
        conn.commit()
        cursor.close()
    finally:
        release_connection(conn)


def save_face_embedding(user_id, embedding):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE "Users" SET "FaceEmbedding" = %s WHERE "Id" = %s',
            (embedding.tolist(), user_id)
        )
        conn.commit()
        cursor.close()
    finally:
        release_connection(conn)


def get_all_embeddings():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT "Id", "FullName", "FaceEmbedding" FROM "Users" '
            'WHERE "FaceEmbedding" IS NOT NULL AND "IsDeleted" = false'
        )
        rows = cursor.fetchall()
        cursor.close()
    finally:
        release_connection(conn)

    result = []
    for row in rows:
        result.append({
            "id": row[0],
            "full_name": row[1],
            "embedding": row[2]
        })
    return result
