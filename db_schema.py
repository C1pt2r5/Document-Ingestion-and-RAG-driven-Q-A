import psycopg
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection string
CONNECTION_STRING = os.environ.get("PG_CONNECTION_STRING", "postgresql://postgres:Mklop9009@@localhost:5432/ragapp")

def init_db():
    """Initialize the database schema"""
    conn = psycopg.connect(CONNECTION_STRING)
    try:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR(36) PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create documents table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id VARCHAR(36) PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            user_id VARCHAR(36) REFERENCES users(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create document_chunks table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id VARCHAR(36) PRIMARY KEY,
            document_id VARCHAR(36) REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create selected_documents table for tracking user document selections
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS selected_documents (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(36) REFERENCES users(id),
            document_id VARCHAR(36) REFERENCES documents(id),
            selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, document_id)
        )
        """)
        
        conn.commit()
        print("Database schema initialized successfully")
    except Exception as e:
        conn.rollback()
        print(f"Error initializing database: {str(e)}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    init_db()