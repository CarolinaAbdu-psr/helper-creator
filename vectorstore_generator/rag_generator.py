import argparse
from pathlib import Path
import os
import shutil

import docs_collector
import rag_utils as rag_utils


def generate_vectorstore_from_path(source_path: str, output_name: str):
    """
    Generate vectorstore from any path.
    
    Args:
        source_path: Path to the directory containing documents
        output_name: Name for the output (used for persist_dir and data_dir naming)
    """
    # Create shared directories
    data_dir = Path("data").resolve()
    chromadb_dir = Path("chromadb").resolve()
    
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print("Removed existing data folder")

    data_dir.mkdir(exist_ok=True)
    chromadb_dir.mkdir(exist_ok=True)
    

    persist_dir = chromadb_dir / output_name
    source_data_dir = data_dir 
    
    # Collect files list
    print(f">> Getting files from: {source_path}")
    files = docs_collector.get_documents(source_path)
    print(f"Found {len(files)} files")
    
    if not files:
        print("No files found. Exiting.")
        return
    
    # Copying Files from docs to data/chromadb directory 
    print(">> Copying files")
    rag_utils.copy_to_data_dir(files, output_name, source_data_dir)
    
    # Loading Files from data/chromadb 
    print(">> Loading files")
    documents = rag_utils.load_documents(source_data_dir)
    
    #Create vectorstore with loaded documents
    print(f">> Creating {output_name} vectorstore")
    rag_utils.create_vectorstore(documents, str(persist_dir))
    
    print(f"Finished creating {output_name} vectorstore in: {persist_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG vectorstore from directory")
    parser.add_argument("--path",default="vectorstore_generator/docs/properties", help="Path to the directory containing documents")
    parser.add_argument("--output",default="chromadb", help="Name for the output vectorstore chromadb/output")
    
    args = parser.parse_args()
    
    try:
        generate_vectorstore_from_path(args.path, args.output)
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


