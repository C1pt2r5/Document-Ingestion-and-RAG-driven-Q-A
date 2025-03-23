import os
from fastapi import FastAPI, HTTPException, Depends, Body
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import psycopg
import uuid

# Import our modules
from document_processor import DocumentProcessor
from rag_service import RAGService

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Document Management and RAG Q&A API")

# Configuration
CONNECTION_STRING = os.environ.get("PG_CONNECTION_STRING", "postgresql://postgres:postgres@localhost:5432/ragapp")
COLLECTION_NAME = "documents"

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize vector store
vector_store = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# Initialize document processor and RAG service
document_processor = DocumentProcessor(vector_store)
rag_service = RAGService(vector_store)

# Database connection
def get_db_connection():
    conn = psycopg.connect(CONNECTION_STRING)
    try:
        yield conn
    finally:
        conn.close()

# Models
class UserBase(BaseModel):
    username: str
    email: str

class DocumentBase(BaseModel):
    title: str
    content: str
    user_id: str
    metadata: Optional[dict] = {}

class QuestionBase(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None

class DocumentSelectionBase(BaseModel):
    document_ids: List[str]

# User API
@app.post("/users/", status_code=201)
async def create_user(user: UserBase, conn=Depends(get_db_connection)):
    user_id = str(uuid.uuid4())
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (id, username, email) VALUES (%s, %s, %s) RETURNING id",
            (user_id, user.username, user.email)
        )
        conn.commit()
        return {"id": user_id, **user.dict()}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        cursor.close()

# Document Ingestion API
@app.post("/documents/", status_code=201)
async def ingest_document(document: DocumentBase):
    try:
        # Process document
        metadata = {
            "title": document.title,
            "user_id": document.user_id,
            **document.metadata
        }
        
        chunk_ids = await document_processor.process_document(document.content, metadata)
        
        # Store document metadata in database
        conn = psycopg.connect(CONNECTION_STRING)
        try:
            cursor = conn.cursor()
            document_id = metadata["document_id"]
            
            cursor.execute(
                "INSERT INTO documents (id, title, user_id) VALUES (%s, %s, %s)",
                (document_id, document.title, document.user_id)
            )
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()
        
        return {
            "status": "success",
            "document_id": metadata["document_id"],
            "chunks_processed": len(chunk_ids),
            "vector_ids": chunk_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during document ingestion: {str(e)}")

# Q&A API
@app.post("/qa/")
async def answer_question(question_data: QuestionBase):
    try:
        # Use RAG service to answer question
        result = await rag_service.answer_question(
            question_data.question,
            document_ids=question_data.document_ids
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during question answering: {str(e)}")

# Document Selection API
@app.post("/documents/select/")
async def select_documents(selection: DocumentSelectionBase, conn=Depends(get_db_connection)):
    try:
        # Validate that all document IDs exist
        cursor = conn.cursor()
        placeholders = ", ".join(["%s"] * len(selection.document_ids))
        cursor.execute(
            f"SELECT id FROM documents WHERE id IN ({placeholders})",
            tuple(selection.document_ids)
        )
        found_ids = [row[0] for row in cursor.fetchall()]
        
        if len(found_ids) != len(selection.document_ids):
            missing_ids = set(selection.document_ids) - set(found_ids)
            raise HTTPException(
                status_code=404,
                detail=f"Documents not found: {', '.join(missing_ids)}"
            )
        
        return {"status": "success", "selected_documents": selection.document_ids}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during document selection: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)