from pathlib import Path
import os
import shutil
from typing import List, Dict, Any


from langchain_community.document_loaders import JSONLoader, TextLoader, PythonLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, PythonCodeTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings
from dotenv import load_dotenv

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# CHROMA DB SETTINGS
#--------------------------------------------------------
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True
)

load_dotenv()

# -------------------------------------------------------
# Copy Raw Data into data directory
#--------------------------------------------------------
def copy_to_data_dir(file_list: List[str], folder_name: str, data_dir: Path):
    """Copy files to data_dir/folder name directory"""
    os.makedirs(data_dir / folder_name, exist_ok=True)
    for file in file_list:
        shutil.copy(file, data_dir / folder_name)


# -------------------------------------------------------
# Load Documents from data/chromadb 
#--------------------------------------------------------

JSON_CONTRASTIVE_JQ_SCHEMA = '.[]'  # iterate over all elements 

def load_documents(data_dir: Path) -> List[Document]:
    """ 
    Loads documents from data_dir and treta specially json files. 
    Saves only the natural question as page content and the rest save it as metadata.
    """
    try:
        all_documents = []
        processed_files = set()
            
        logger.info(f"Data directory: {data_dir}")
        
        if not data_dir.exists():
            raise ValueError(f"Directory not found: {data_dir}")
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                str_path = str(Path(root) / file)
                
                if str_path in processed_files:
                    continue
                    
                try:
                    file_extension = Path(str_path).suffix.lower()
                   
                    if file_extension == '.json' and  str_path.endswith("examples.json"):
                        
                        loader = JSONLoader(
                            file_path=str_path,
                            jq_schema=JSON_CONTRASTIVE_JQ_SCHEMA,
                            text_content=False, 
                            metadata_func=lambda obj, metadata: {
                                "source": metadata["source"],
                                "natural_request": obj.get("natural_request"),
                                "correct_code": obj.get("correct_code").get("code"),
                                "correct_inst": obj.get("correct_code").get("instruction"),
                                "incorrect_code": obj.get("incorrect_code").get("code"),
                                "incorrect_inst": obj.get("incorrect_code").get("error")
                            }
                        )
                    
                        docs = loader.load()
                        for doc in docs:
                            #Converts only the natural request to vector 
                            doc.page_content = doc.metadata.pop("natural_request")
                            
                    elif file_extension == '.json' and str_path.endswith("properties.json"):
                        def safe_to_string(value):
                            """Convert list to string to save it on vetorstore"""
                            if isinstance(value, list):
                                return ", ".join(map(str, value))
                            return value

                        loader = JSONLoader(
                            file_path=str_path,
                            jq_schema='.[]' ,
                            text_content=False, 
                            metadata_func=lambda obj, metadata: {
                                "source": metadata["source"],
                                "object": obj.get("object"),
                                "mandatory": safe_to_string(obj.get("properties").get("mandatory")),
                                "references_objects": safe_to_string(obj.get("properties").get("references_objects")),
                                "static_properties": safe_to_string(obj.get("properties").get("static_properties")),
                                "dynamic_properties": safe_to_string(obj.get("properties").get("dynamic_properties"))
                            }
                        )

                        docs = loader.load()
                        for doc in docs:
                            #Converts only the natural question to vector 
                            doc.page_content = doc.metadata.pop("object")

                    else:
                        loader = TextLoader(str_path)
                    
                    if file_extension != '.json':
                        docs = loader.load()

                    all_documents.extend(docs)
                    processed_files.add(str_path)
                    logger.info(f"Loaded: {str_path} ({len(docs)} documents/chunks)")
                
                except Exception as e:
                    logger.warning(f"Error loading {str_path}: {str(e)}")
                    continue

        logger.info(f"Total precessed files: {len(processed_files)}")
        for f in sorted(processed_files):
            logger.info(f"  - {f}")
        
        return all_documents
    
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise

# -------------------------------------------------------
# Use embedding model to create vectorstore
#--------------------------------------------------------

def create_vectorstore(documents: List[Document], persist_directory: str) -> Chroma:
    """
    Transform all loaded documents in embeeddings. 
    Only the natural question from json files will be transformed in embeddings.
    """

    if not documents:
        logger.warning("No documents provided. Cannot create vectorstore.")
        return None
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Divide text files into smaller texts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    final_chunks = []

    # Add natural question as a unique chunck, unless is too long
    for doc in documents:
        metadata = doc.metadata
        source = metadata.get('source', '')

        if source.endswith('.json'):
            if len(doc.page_content) > 700: 
                logger.warning(f"Json question to large: {len(doc.page_content)} chars. Spliting...")
                chunks = text_splitter.split_documents([doc])
                final_chunks.extend(chunks)
            else:
                final_chunks.append(doc) #Add a question as a chunk 
            continue
    
    # Check chunk size limits 
    max_chunk_size = 7000
    filtered_chunks = []
    skipped_count = 0
    
    for chunk in final_chunks:
        if len(chunk.page_content) <= max_chunk_size:
            filtered_chunks.append(chunk)
        else:
            skipped_count += 1
            logger.warning(f"Skipping chunk of size {len(chunk.page_content)} characters (too large)")
            
            # Try to split large chunks further
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            sub_chunks = sub_splitter.split_documents([chunk])
            
            for sub_chunk in sub_chunks:
                if len(sub_chunk.page_content) <= max_chunk_size:
                    filtered_chunks.append(sub_chunk)
                else:
                    logger.warning(f"Skipping sub-chunk of size {len(sub_chunk.page_content)} characters (still too large)")
    
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} chunks that were too large")
    
    logger.info(f"Final chunk count: {len(filtered_chunks)}")
    
    if not filtered_chunks:
        logger.warning("No valid chunks created from documents. Cannot create vectorstore.")
        return None

    # Process in smaller batches to avoid OpenAI token limits
    batch_size = 50  # Process 50 chunks at a time (conservative for token limits)
    
    # Create initial empty vectorstore
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    
    vectorstore = None
    
    for i in range(0, len(filtered_chunks), batch_size):
        batch = filtered_chunks[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(filtered_chunks) + batch_size - 1)//batch_size} ({len(batch)} chunks)")
        
        if vectorstore is None:
            # Create initial vectorstore
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
                client_settings=CHROMA_SETTINGS
            )
        else:
            # Add to existing vectorstore
            vectorstore.add_documents(batch)
    
    return vectorstore 