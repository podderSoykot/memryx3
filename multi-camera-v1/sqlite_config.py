import sqlite3
import os
from contextlib import contextmanager

class SQLiteDB:
    def __init__(self, db_path="faces.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with optimized settings"""
        if not os.path.exists(self.db_path):
            print(f"Creating new SQLite database: {self.db_path}")
            self.create_tables()
        else:
            print(f"Using existing SQLite database: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with optimizations"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        
        # SQLite performance optimizations
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        conn.execute("PRAGMA cache_size=10000")  # Larger cache
        conn.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory map
        
        try:
            yield conn
        finally:
            conn.close()
    
    def create_tables(self):
        """Create all required tables with proper indexes"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Records table with index
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                model TEXT NOT NULL,
                detector TEXT NOT NULL,
                last_backup_timestamp INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Face embeddings table with index
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding_data BLOB NOT NULL,
                identity TEXT NOT NULL,
                model TEXT NOT NULL,
                detector TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                apparent_age TEXT DEFAULT 'Unknown',
                dominant_gender TEXT DEFAULT 'Unknown',
                last_modified INTEGER,
                normalization TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Pictures table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS pictures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                file_path TEXT,
                last_modified INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Backup metadata table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS backup_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT,
                last_backup_timestamp INTEGER NOT NULL,
                table_name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_identity ON records(identity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_timestamp ON records(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_face_embeddings_identity ON face_embeddings(identity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_face_embeddings_timestamp ON face_embeddings(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pictures_timestamp ON pictures(timestamp)")
            
            conn.commit()
            print("SQLite database tables created successfully with optimizations!")
    
    def save_face_embedding(self, embedding, identity, model, detector, timestamp, age="Unknown", gender="Unknown", normalization="base"):
        """Save face embedding to database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            import pickle
            import numpy as np
            
            # Serialize embedding
            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], dict):
                # Handle DeepFace result format with dictionaries
                actual_embedding = np.array(embedding[0]['embedding'])
            else:
                # Handle direct embedding array or list
                actual_embedding = np.array(embedding)
            
            serialized_embedding = pickle.dumps(actual_embedding)
            
            cursor.execute("""
                INSERT INTO face_embeddings 
                (embedding_data, identity, model, detector, timestamp, apparent_age, dominant_gender, normalization, last_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (serialized_embedding, identity, model, detector, timestamp, str(age), str(gender), normalization, timestamp))
            
            conn.commit()
            return cursor.lastrowid
    
    def save_detection_record(self, identity, model, detector, timestamp):
        """Save detection record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO records (identity, model, detector, timestamp)
                VALUES (?, ?, ?, ?)
            """, (identity, model, detector, timestamp))
            conn.commit()
            return cursor.lastrowid
    
    def save_picture(self, file_name, epoch_time, file_path):
        """Save picture information to database"""
        try:
            # Check if file_path column exists, if not add it
            cursor = self.connection.cursor()
            cursor.execute("PRAGMA table_info(pictures)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'file_path' not in columns:
                cursor.execute("ALTER TABLE pictures ADD COLUMN file_path TEXT")
                self.connection.commit()
                print("📁 Added file_path column to pictures table")
            
            # Now save the picture with file_path
            cursor.execute(
                "INSERT INTO pictures (file_name, epoch_time, file_path) VALUES (?, ?, ?)",
                (file_name, epoch_time, file_path)
            )
            self.connection.commit()
            print(f"📷 Picture saved: {file_name}")
            
        except Exception as e:
            print(f"❌ Failed to save picture: {e}")
    
    def get_face_count(self):
        """Get total number of unique faces"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT identity) FROM face_embeddings")
            return cursor.fetchone()[0]
    
    def get_detection_count(self):
        """Get total number of detections"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM records")
            return cursor.fetchone()[0]
    
    def get_recent_detections(self, limit=10):
        """Get recent detections"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT identity, timestamp, model, detector 
                FROM records 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            return cursor.fetchall()
    
    def get_all_embeddings(self):
        """Get all face embeddings for comparison"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT identity, embedding_data, apparent_age, dominant_gender, timestamp
                FROM face_embeddings
            """)
            results = cursor.fetchall()
            
            embeddings = []
            for identity, embedding_data, age, gender, timestamp in results:
                import pickle
                embedding = pickle.loads(embedding_data)
                embeddings.append({
                    'identity': identity,
                    'embedding': embedding,
                    'age': age,
                    'gender': gender,
                    'timestamp': timestamp
                })
            return embeddings
    
    def cleanup_old_records(self, days=30):
        """Cleanup old records older than specified days"""
        import time
        cutoff_timestamp = int(time.time()) - (days * 24 * 60 * 60)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM records WHERE timestamp < ?", (cutoff_timestamp,))
            deleted_count = cursor.rowcount
            conn.commit()
            print(f"Cleaned up {deleted_count} old records")
            return deleted_count 