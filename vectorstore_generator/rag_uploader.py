import datetime as dt
import zipfile
import os

import api_s3


"""This code uploded created vectorstore into S3 api"""

def zip_folder_contents(folder_path: str, zip_name: str) -> str:
    """Zip folder to upload"""
    try:
        # Create vectorstore directory if it doesn't exist
        vectorstore_dir = "vectorstore"
        os.makedirs(vectorstore_dir, exist_ok=True)
        
        # Save ZIP file inside vectorstore directory
        zip_path = os.path.join(vectorstore_dir, zip_name)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # zip all files in the folder
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))
        print(f"Contents of {folder_path} zipped into {zip_path}")
        return zip_path
    except Exception as e:
        print(f"Error zipping folder contents: {str(e)}")
        raise


def find_rag_directory() -> str:

    # New structure: chromadb/<name>/
    chromadb_path = os.path.join("chromadb")
    
    if os.path.exists(chromadb_path) and os.path.isdir(chromadb_path):
        return chromadb_path
    
    # Fallback to old structures for backward compatibility
    old_structures = [
        f"rag/vectorstore",  # Previous structure
        f"chroma_db"        # Original structure
    ]
    
    for directory in old_structures:
        if os.path.exists(directory) and os.path.isdir(directory):
            return directory
    
    # List available directories to help user
    available_dirs = []
    if os.path.exists("chromadb"):
        available_dirs.extend([f"chromadb/{d}" for d in os.listdir("chromadb") if os.path.isdir(os.path.join("chromadb", d))])
    
    available_dirs.extend([d for d in os.listdir('.') if (d.startswith('rag_') or d.startswith('chroma_db')) and os.path.isdir(d)])
    
    if available_dirs:
        raise ValueError(f"Directory not found. Available RAG directories: {', '.join(available_dirs)}")
    else:
        raise ValueError(f"No RAG directories found. Please generate a RAG database first.")

def upload_rag():
    directory = find_rag_directory()
    name = f"rag_cypher_examples{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"
    
    print(f"Generated RAG file name: {name}")
    print(f"Using directory: {directory}")
    zip_path = zip_folder_contents(directory, name)

    # upload to S3
    try:
        api_s3.upload_file_to_s3(zip_path, name)
        print(f"File {name} uploaded to S3 successfully.")
    except Exception as e:
        print(f"Error uploading file to S3: {str(e)}")
        raise


if __name__ == "__main__":
    import sys


    if len(sys.argv) > 2:
        print("Usage: python rag_uploader.py")
        print("Examples: factory, knowledge_base, or any custom name")
        print("If no argument is provided, defaults to factory")
        exit(1)
    
    try:
        upload_rag()
    except ValueError as e:
        print(f"Error: {e}")
        print("Usage: python rag_uploader.py")
        exit(1)