"""LLM processors for query refinement and RAG operations."""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import torch
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

from ..utils.logging import get_logger
from ..utils.metrics import track_performance
from ..vectorstore.store import VectorStoreManager
from config.settings import (
    DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMPERATURE, DEFAULT_API_KEY,
    DEFAULT_CHUNK_SIZE, USE_GPU
)

logger = get_logger(__name__)

class QueryRefiner:
    """Class to handle LLM-based query refinement with GPU awareness."""
    
    def __init__(self, 
                model_name: str = DEFAULT_LLM_MODEL, 
                temperature: float = 0.0, 
                api_key: str = DEFAULT_API_KEY,
                cache_dir: str = "query_cache"):
        """
        Initialize the query refiner with settings and caching.
        
        Args:
            model_name: Name of the LLM model
            temperature: Temperature for generation
            api_key: API key for the LLM service
            cache_dir: Directory for caching results
        """
        self.model_name = model_name
        self.temperature = temperature
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.chat = ChatGroq(
            temperature=temperature, 
            model_name=model_name, 
            api_key=api_key
        )
        
        # Define the system prompt
        self.system_prompt = """You are an expert Search Query Optimizer. Your sole purpose is to rewrite user search queries to improve their effectiveness in retrieving relevant results from search engines like DuckDuckGo.

GUIDELINES:
- Maintain the original user intent and core meaning
- Remove unnecessary filler words (like "please find" or "I want to know about")
- Add relevant synonyms or alternative phrasings when helpful
- Include important context terms that might be implied but not stated
- Break down complex queries into clearer structures
- Correct misspellings and grammatical errors
- Avoid introducing personal opinions or biases
- Never add content warnings or explanations - return only the optimized query
- Format the output as plain text with no quotes or decorations

EXAMPLES:
Original: "how do i make my computer faster it's really slow lately"
Optimized: computer performance optimization slow PC troubleshooting

Original: "places to eat near times square with kids friendly"
Optimized: family-friendly restaurants times square kid-friendly dining

Original: "tesla model 3 vs model y which is better to buy"
Optimized: tesla model 3 model y comparison pros cons buying guide

Remember: Your output should ONLY be the rewritten search query, nothing else."""

        logger.info(f"Initialized query refiner with model: {model_name}")
    
    def _get_cache_path(self, query: str) -> str:
        """
        Get cache file path for a query.
        
        Args:
            query: Original query
            
        Returns:
            Cache file path
        """
        # Create a safe filename from the query
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{query_hash}.json")
    
    def _check_cache(self, query: str) -> Optional[str]:
        """
        Check if a query result is in cache.
        
        Args:
            query: Original query
            
        Returns:
            Cached refined query if exists, None otherwise
        """
        cache_path = self._get_cache_path(query)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                return cache_data.get('refined_query')
            except Exception as e:
                logger.warning(f"Error reading cache: {str(e)}")
        return None
    
    def _save_cache(self, query: str, refined_query: str) -> None:
        """
        Save a refined query to cache.
        
        Args:
            query: Original query
            refined_query: Refined query
        """
        cache_path = self._get_cache_path(query)
        try:
            cache_data = {
                'original_query': query,
                'refined_query': refined_query,
                'timestamp': datetime.now().isoformat()
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving cache: {str(e)}")
    
    @track_performance
    def rewrite_query(self, user_query: str, use_cache: bool = True) -> str:
        """
        Rewrite a user query to optimize for search engines with caching.
        
        Args:
            user_query: Original user query
            use_cache: Whether to use cache
            
        Returns:
            Refined query
        """
        # Check cache first if enabled
        if use_cache:
            cached_query = self._check_cache(user_query)
            if cached_query:
                logger.info(f"Using cached refinement for query: {user_query}")
                return cached_query
        
        try:
            # Create the prompt template
            human = "{text}"
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", human)
            ])
            
            # Create and execute the chain
            chain = prompt | self.chat
            
            # Get the refined query
            refined_query = chain.invoke({"text": user_query}).content
            
            # Save to cache if enabled
            if use_cache:
                self._save_cache(user_query, refined_query)
            
            logger.info(f"Refined query: '{user_query}' -> '{refined_query}'")
            return refined_query
        except Exception as e:
            logger.error(f"Error refining query: {str(e)}")
            return user_query  # Return original query on error


class RAGQueryProcessor:
    """GPU-aware class to handle RAG-based query processing with optimized performance."""
    
    def __init__(self, 
                 vector_store_manager: VectorStoreManager,
                 llm_model: str = DEFAULT_LLM_MODEL,
                 temperature: float = DEFAULT_LLM_TEMPERATURE,
                 api_key: str = DEFAULT_API_KEY,
                 max_tokens_per_chunk: int = DEFAULT_CHUNK_SIZE):
        """
        Initialize the RAG processor.
        
        Args:
            vector_store_manager: Vector store manager
            llm_model: LLM model name
            temperature: Temperature for generation
            api_key: API key for the LLM service
            max_tokens_per_chunk: Maximum tokens per chunk
        """
        self.vector_store_manager = vector_store_manager
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens_per_chunk = max_tokens_per_chunk
        
        # Initialize LLM with optimal settings
        self.llm = ChatGroq(
            temperature=temperature,
            model_name=llm_model,
            api_key=api_key,
            max_tokens=max_tokens_per_chunk
        )
        
        # Initialize the memory saver for checkpointing
        self.memory = MemorySaver()
        
        # Set up the retrieval prompt with better context handling
        self.retrieval_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing search results and providing accurate, well-reasoned answers.
            Your task is to analyze the provided context carefully and answer the user's question.
            
            Guidelines:
            - Focus on information directly present in the context
            - Be concise but comprehensive
            - If the context doesn't contain enough information to answer fully, acknowledge this
            - If you don't know the answer, just say so - don't make up information
            - Cite your sources when appropriate
            - Prioritize recent and authoritative information
            - Organize your response in a clear, logical structure
            - Use bullet points for lists when appropriate"""),
            ("human", """Context:
{context}

Question: {question}

Please provide a well-structured answer based on the context. If the context doesn't contain enough information to answer the question fully, note this limitation.""")
        ])
        
        # Create the optimized retrieval chain
        self.retrieval_chain = (
            {"context": self.retrieve, "question": RunnablePassthrough()}
            | self.retrieval_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Create the agent executor with the decorated retrieve tool
        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=[self._get_retrieve_tool()],
            checkpointer=self.memory
        )
        
        logger.info(f"Initialized RAG processor with model: {llm_model}")
    
    @staticmethod
    def _chunk_text(text: str, max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """
        Split text into chunks of appropriate size for LLM processing.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum size of each chunk in characters
            
        Returns:
            List of text chunks
        """
        # If text is small enough, return as is
        if len(text) <= max_chunk_size:
            return [text]
        
        # Split into paragraphs first for better semantic boundaries
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If paragraph is too large on its own, split it further
            if para_size > max_chunk_size:
                sentences = para.split('. ')
                for sentence in sentences:
                    if current_size + len(sentence) > max_chunk_size and current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = [sentence]
                        current_size = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_size += len(sentence)
            elif current_size + para_size > max_chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _get_retrieve_tool(self):
        """
        Create and return the retrieve tool with proper decoration.
        
        Returns:
            Decorated retrieve tool
        """
        @tool(response_format="content_and_artifact")
        def retrieve(query: str, k: int = 5) -> Tuple[str, List[Document]]:
            """
            Retrieve relevant documents from the vector store.
            
            Args:
                query: The search query
                k: Number of documents to retrieve
                
            Returns:
                Tuple of (formatted string of results, list of retrieved documents)
            """
            docs = self.vector_store_manager.search(query, k=k)
            # Format documents with truncated content to manage token size
            formatted_docs = "\n\n".join(
                f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content[:1000]}..."
                for doc in docs
            )
            return formatted_docs, docs
        
        return retrieve
    
    @track_performance
    def retrieve(self, query: str, k: int = 5) -> str:
        """
        Retrieve and format relevant documents for the retrieval chain.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            Formatted context string
        """
        docs = self.vector_store_manager.search(query, k=k)
        
        # Format documents with source information
        all_text = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            text = f"[Source {i+1}: {source}]\n{doc.page_content}"
            all_text.append(text)
        
        combined_text = "\n\n".join(all_text)
        
        # Chunk the text if it's too large
        chunks = self._chunk_text(combined_text)
        
        # Use the first (most relevant) chunk to stay within token limits
        return chunks[0] if chunks else ""
    
    @track_performance
    def process_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Process a query using RAG and return refined results.
        
        Args:
            query: The user's question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing the answer and supporting information
        """
        try:
            # Get the refined answer using the retrieval chain
            answer = self.retrieval_chain.invoke(query)
            
            # Get the agent's reasoning using the executor with reduced context
            agent_response = self.agent_executor.invoke({
                "input": f"Analyze this question concisely: {query}",
                "chat_history": []
            })
            
            return {
                "query": query,
                "answer": answer if isinstance(answer, str) else (answer.content if hasattr(answer, 'content') else str(answer)),
                "agent_reasoning": agent_response.content if hasattr(agent_response, 'content') else str(agent_response),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # Attempt to process with reduced context
            try:
                # Fallback to simpler processing
                docs = self.vector_store_manager.search(query, k=2)  # Reduce number of documents
                context = "\n".join(doc.page_content[:800] for doc in docs)  # Limit content size
                
                response = self.llm.invoke(
                    f"Context: {context}\n\nQuestion: {query}\n\nProvide a concise answer:"
                )
                
                return {
                    "query": query,
                    "answer": response.content if hasattr(response, 'content') else str(response),
                    "agent_reasoning": "Fallback processing due to token limit or error",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e2:
                return {
                    "query": query,
                    "error": f"Primary error: {str(e)}\nFallback error: {str(e2)}",
                    "timestamp": datetime.now().isoformat()
                }