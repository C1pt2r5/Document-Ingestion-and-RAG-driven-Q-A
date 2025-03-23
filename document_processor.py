from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
import asyncio
from typing import List, Dict, Any

class DocumentProcessor:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
    async def process_document(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Process a document by splitting it into chunks and generating embeddings
        
        Args:
            content: The document content
            metadata: Document metadata including title, user_id, etc.
            
        Returns:
            List of document chunk IDs
        """
        # Generate a document ID if not provided
        document_id = metadata.get("document_id", str(uuid.uuid4()))
        if "document_id" not in metadata:
            metadata["document_id"] = document_id
            
        # Split document into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Create Document objects with metadata
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_id": i,
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        # Add documents to vector store
        chunk_ids = await asyncio.to_thread(
            self.vector_store.add_documents,
            docs
        )
        
        return chunk_ids
        
    async def batch_process_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Process multiple documents in batch
        
        Args:
            documents: List of document dictionaries with content and metadata
            
        Returns:
            Dictionary mapping document IDs to lists of chunk IDs
        """
        results = {}
        
        for doc in documents:
            content = doc.pop("content")
            metadata = doc
            
            document_id = metadata.get("document_id", str(uuid.uuid4()))
            if "document_id" not in metadata:
                metadata["document_id"] = document_id
                
            chunk_ids = await self.process_document(content, metadata)
            results[document_id] = chunk_ids
            
        return results