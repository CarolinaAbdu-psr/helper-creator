import argparse
import glob
import os
import subprocess
from typing import List


def get_documents(base_path: str) -> List[str]:
    """
    Generic function to collect documents from any path.
    
    Args:
        base_path: Path to the directory to search
    
    Returns:
        List of file paths found
    """
    basename_filter = [
        "__init__.py",
        ".gitignore"
    ]
    
    documents = []
    
    if not os.path.exists(base_path):
        print(f"Warning: Path {base_path} does not exist")
        return documents
    
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file in basename_filter or file.startswith('.'):
                continue
                
            file_path = os.path.join(root, file)
            if file.endswith(('.md', '.py','.json')):
                documents.append(file_path)
    
    return sorted(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect documents from directory")
    parser.add_argument("path", help="Path to the directory to collect documents from")
    
    args = parser.parse_args()
    
    print(f">> Getting documents from: {args.path}")
    documents = get_documents(args.path)
    print(f"Found {len(documents)} files")
    
    print(f"\nDocuments found:")
    for doc in documents:
        print(f"  - {doc}") 