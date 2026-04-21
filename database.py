import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import os
import numpy as np
from datetime import datetime, timezone

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
connection_pool = None


def init_pool():
    global connection_pool
    connection_pool = pool.SimpleConnectionPool(2, 20, DATABASE_URL)  # max 20'ye çıkardık
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

    return [{"id": r[0], "full_name": r[1], "username": r[2]} for r in rows]


def save_attendance(user_id, event_type, shift, description, custom_time=None):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        zaman = custom_time if custom_time else datetime.utcnow()
        cursor.execute(
            'INSERT INTO "Attendances" ("UserId", "Type", "Shift", "Time", "IsLate", "LateReason") '
            'VALUES (%s, %s, %s, %s, %s, %s)',
            (user_id, event_type, shift, zaman, False, description)
        )
        conn.commit()
        cursor.close()
    finally:
        release_connection(conn)  # ← her zaman geri döndür


def save_face_embedding(user_id, embedding):
    conn = get_connection()
    try:                          # ← try/finally ekledik
        cursor = conn.cursor()
        norm = np.linalg.norm(embedding)
        normalized = (embedding / norm).tolist()
        cursor.execute(
            'UPDATE "Users" SET "FaceEmbedding" = %s WHERE "Id" = %s',
            (normalized, user_id)
        )
        conn.commit()
        cursor.close()
    finally:
        release_connection(conn)  # ← conn.close() değil, pool'a geri ver


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

    return [{"id": r[0], "full_name": r[1], "embedding": r[2]} for r in rows]
