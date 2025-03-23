from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional
import asyncio

class RAGService:
    def __init__(self, vector_store, model_name="HuggingFaceH4/zephyr-7b-beta"):
        self.vector_store = vector_store
        
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            temperature=0.7,
        )
        
    async def answer_question(
        self, 
        question: str, 
        document_ids: Optional[List[str]] = None,
        k: int = 3
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG approach
        
        Args:
            question: The user's question
            document_ids: Optional list of document IDs to restrict search
            k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and source information
        """
        # Retrieve relevant documents
        if document_ids:
            # Filter by document IDs if provided
            filter_dict = {"document_id": {"$in": document_ids}}
            docs = await asyncio.to_thread(
                self.vector_store.similarity_search,
                question,
                k=k,
                filter=filter_dict
            )
        else:
            # Otherwise search across all documents
            docs = await asyncio.to_thread(
                self.vector_store.similarity_search,
                question,
                k=k
            )
        
        if not docs:
            return {
                "answer": "No relevant documents found to answer your question.",
                "sources": []
            }
        
        # Construct context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer using RAG approach
        prompt = f"""
        Answer the following question based on the provided context:
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        # Generate response
        response = await asyncio.to_thread(
            self.llm_pipeline,
            prompt
        )
        
        # Extract the answer part (after "Answer:")
        answer = response[0]["generated_text"].split("Answer:")[-1].strip()
        
        return {
            "answer": answer,
            "sources": [
                {
                    "document_id": doc.metadata.get("document_id"),
                    "title": doc.metadata.get("title"),
                    "chunk_id": doc.metadata.get("chunk_id")
                }
                for doc in docs
            ]
        }