import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from core.config import VLLM_EMBED_MODEL, EMBED_BATCH, VLLM_EMBEDDINGS_URL
from core.dependencies import VLLMEmbeddings, get_vllm_chat_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdf(file_path: str):
    logger.info(f"Loading PDF from: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found at path: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    total_pages = len(documents)
    total_content = sum(len(doc.page_content) for doc in documents)
    logger.info(f"Loaded PDF with {total_pages} pages and {total_content} characters")
    
    # CHANGED: Smaller chunks to avoid token limits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(split_docs)} chunks for processing")
    
    return split_docs

def store_documents_for_file(documents, graph, original_filename: str = "Unknown File"):
    logger.info(f"Starting ingestion process for {len(documents)} documents from '{original_filename}'")
    
    llm = get_vllm_chat_llm()
    llm_transformer_filtered = LLMGraphTransformer(
        llm=llm,
        # max_output_tokens=800,  
        node_properties=False,  
    )
    
    batch_size = 2
    processed_docs_count = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        current_batch_num = i // batch_size + 1
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        logger.info(
            f"Processing batch {current_batch_num}/{total_batches} for '{original_filename}' "
            f"(docs {i+1}-{min(i+batch_size, len(documents))}/{len(documents)})"
        )
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                filtered_batch = [doc for doc in batch if len(doc.page_content.strip()) > 50]
                
                if not filtered_batch:
                    logger.warning(f"Batch {current_batch_num} for '{original_filename}' has no valid documents.")
                    break
                
                graph_documents = llm_transformer_filtered.convert_to_graph_documents(filtered_batch)
                
                if graph_documents:
                    graph.add_graph_documents(
                        graph_documents,
                        baseEntityLabel=True,
                        include_source=True
                    )
                    processed_docs_count += len(filtered_batch)
                    logger.info(f"Added batch {current_batch_num} for '{original_filename}' to the graph.")
                    break  
                else:
                    logger.warning(f"Batch {current_batch_num} for '{original_filename}' produced no graph documents.")
                    break 
            
            except Exception as e:
                if "LengthFinishReasonError" in str(e) or "token" in str(e).lower():
                    logger.warning(f"Token limit hit in batch {current_batch_num}, attempt {attempt + 1}. Error: {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2 
                        continue
                    else:
                        logger.error(f"Max retries reached for batch {current_batch_num} due to token limits.")
                        break
                else:
                    logger.error(f"Error processing batch {current_batch_num} for '{original_filename}': {e}", exc_info=True)
                    break
    
    logger.info(f"Successfully added {processed_docs_count} documents from '{original_filename}' to the graph.")
    return processed_docs_count

def get_vllm_embeddings():
    return VLLMEmbeddings(
        base_url=VLLM_EMBEDDINGS_URL,
        model=VLLM_EMBED_MODEL,
        batch_size=EMBED_BATCH * 2 
    )

def test_vllm_services():
    logger.info("Testing vLLM services...")
    
    try:
        embeddings = get_vllm_embeddings()
        test_embedding = embeddings.embed_query("test query")
        logger.info(f"✓ Embeddings service working. Dimension: {len(test_embedding)}")
    except Exception as e:
        logger.error(f"✗ Embeddings service failed: {e}")
        return False
    
    try:
        llm = get_vllm_chat_llm()
        response = llm.invoke("Say 'test successful'")
        logger.info(f"✓ Chat service working. Response: {response.content[:50]}...")
    except Exception as e:
        logger.error(f"✗ Chat service failed: {e}")
        return False
    
    logger.info("✓ All vLLM services are working!")
    return True