import sqlite3
import pandas as pd
import json
import os
import io
import hashlib
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="cognitron_x.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Users Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Datasets Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT NOT NULL,
                    name TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_email) REFERENCES users (email)
                )
            """)

            # Performance Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_user ON datasets(user_email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_sessions(user_email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_sessions(session_id)")
            
            conn.commit()

    # --- User Management ---
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def add_user(self, email, password, role):
        password_hash = self.hash_password(password)
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)",
                    (email, password_hash, role)
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def authenticate_user(self, email, password):
        password_hash = self.hash_password(password)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role FROM users WHERE email = ? AND password_hash = ?",
                (email, password_hash)
            )
            result = cursor.fetchone()
            return result[0] if result else None

    # --- Dataset Management ---
    def save_dataset(self, user_email, name, df):
        data_json = df.to_json(orient='split')
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO datasets (user_email, name, data_json) VALUES (?, ?, ?)",
                (user_email, name, data_json)
            )
            conn.commit()

    def get_datasets(self, user_email):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, created_at FROM datasets WHERE user_email = ? ORDER BY created_at DESC",
                (user_email,)
            )
            return cursor.fetchall()

    def load_dataset(self, dataset_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data_json FROM datasets WHERE id = ?", (dataset_id,))
            result = cursor.fetchone()
            if result:
                return pd.read_json(io.StringIO(result[0]), orient='split')
            return None

    # --- Chat History Management ---
    def save_chat_history(self, user_email, session_id, messages):
        messages_json = json.dumps(messages)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Check if session exists
            cursor.execute(
                "SELECT id FROM chat_sessions WHERE user_email = ? AND session_id = ?",
                (user_email, session_id)
            )
            exists = cursor.fetchone()
            
            if exists:
                cursor.execute(
                    "UPDATE chat_sessions SET messages_json = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                    (messages_json, exists[0])
                )
            else:
                cursor.execute(
                    "INSERT INTO chat_sessions (user_email, session_id, messages_json) VALUES (?, ?, ?)",
                    (user_email, session_id, messages_json)
                )
            conn.commit()

    def load_chat_history(self, user_email, session_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT messages_json FROM chat_sessions WHERE user_email = ? AND session_id = ?",
                (user_email, session_id)
            )
            result = cursor.fetchone()
            return json.loads(result[0]) if result else []

# Global instance
db = DatabaseManager()
