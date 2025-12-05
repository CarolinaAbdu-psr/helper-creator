import logging
import os
import zipfile

from vectorstore_generator import api_s3

import datetime as dt
from typing import (
    Tuple,
    List,
    Annotated,
    Dict,
    Any
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_vectorstore_directory(doc_type: str) -> str:
    return f'vectorstores/{doc_type}'

def get_rag_list() -> List[str]:
    """Get list of all vectorstores"""
    try:
        rag_list = api_s3.list_files_in_s3()
        return rag_list
    except Exception as e:
        logger.error(f"Error getting RAG list from S3: {str(e)}")
        return []
    

def get_rag_list_with_dates(rag_list: List[str], source_type: str = None) -> List[Tuple[dt.datetime, str]]:
    """Get a list sorted by date of the available rags of a source type: kwnoledge_hub, factory, psrio"""
    # name format: rag_{source_type}_{YYYY-MM-DD_HH-MM-SS}.zip
    rag_with_dates = []
    for rag in rag_list:
        try:
            if not rag.startswith('rag_') or not rag.endswith('.zip'):
                continue
            rag_basename = rag.replace('.zip', '')
            
            # Check if this is old format first (rag_YYYY-MM-DD_HH-MM-SS)
            splitted_name = rag_basename.split('_')
            if len(splitted_name) == 3 and len(splitted_name[1]) == 10 and '-' in splitted_name[1]:
                # Old format: rag_YYYY-MM-DD_HH-MM-SS
                if source_type is not None:
                    continue  # Skip old format files when filtering by source_type
                date_str = splitted_name[1] + ' ' + splitted_name[2].replace('-', ':')
                date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                rag_with_dates.append((date_obj, rag))
                continue
            
            # New format: rag_{source_type}_{YYYY-MM-DD}_{HH-MM-SS}
            # Find the last two parts that look like date and time
            date_part = None
            time_part = None
            extracted_source_type = None
            
            if len(splitted_name) >= 4:
                # Try to find date pattern (YYYY-MM-DD) and time pattern (HH-MM-SS)
                for i in range(len(splitted_name) - 1, 0, -1):
                    part = splitted_name[i]
                    if len(part) == 8 and part.count('-') == 2:  # HH-MM-SS
                        time_part = part
                        if i > 0 and len(splitted_name[i-1]) == 10 and splitted_name[i-1].count('-') == 2:  # YYYY-MM-DD
                            date_part = splitted_name[i-1]
                            # Everything between 'rag' and date_part is the source_type
                            extracted_source_type = '_'.join(splitted_name[1:i-1])
                            break
            
            if not (date_part and time_part and extracted_source_type):
                continue
                
            # If source_type is specified, filter by it
            if source_type and extracted_source_type != source_type:
                continue
                
            date_str = date_part + ' ' + time_part.replace('-', ':')
            date_obj = dt.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            rag_with_dates.append((date_obj, rag))
            
        except Exception as e:
            logger.error(f"Error processing RAG {rag}: {str(e)}")
            continue
    return sorted(rag_with_dates, key=lambda x: x[0], reverse=True)


def extract_rag_to_folder(rag_name: str, folder_path: str) -> bool:
    """Extract vectorstore from s3 to vectorstores/source_type folder"""
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        zip_path = os.path.join(folder_path, rag_name)
        api_s3.download_file_from_s3(rag_name, zip_path)

        logger.info(f"RAG {rag_name} downloaded to {folder_path}")
        if not zip_path.endswith('.zip'):
            zip_path += '.zip'
        logger.info(f"Extracting RAG {rag_name} from {zip_path} to {folder_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
            logger.info(f"RAG {rag_name} extracted to {folder_path}")
        # Remove the zip file after extraction
        os.remove(zip_path)
        return True
    except Exception as e:
        logger.error(f"Error extracting RAG {rag_name}: {str(e)}")
        return False


def get_latest_rag_date(source_type: str = None) -> dt.datetime:
    """Returns the date of the last vectorstore of a source type: knowledge_base, facory, psrio"""
    rag_list = get_rag_list()
    rag_with_dates = get_rag_list_with_dates(rag_list, source_type)

    if rag_with_dates:
        date, _ = rag_with_dates[0]
        return date
    else:
        logger.warning(f"No RAG files found for source_type: {source_type}")
        raise ValueError(f"No RAG files found for source_type: {source_type}")


def download_rag(rag_name: str, chroma_dir_name: str, source_type: str = None) -> dt.datetime:
    """Download a sspecific rag vectorstore to local vectorstore folder"""
    rag_list = get_rag_list()
    if rag_name not in rag_list:
        logger.error(f"RAG {rag_name} not found in the available RAG list.")
        raise ValueError(f"RAG {rag_name} not found in the available RAG list.")

    rag_with_dates = get_rag_list_with_dates(rag_list, source_type)
    for date, rag in rag_with_dates:
        if rag == rag_name:
            extract_rag_to_folder(rag, chroma_dir_name)
            return date

    logger.error(f"RAG {rag_name} not found in the RAG list with dates.")
    raise ValueError(f"RAG {rag_name} not found in the RAG list with dates.")


def download_latest_rag(chroma_dir_name: str, source_type: str = "properties") -> dt.datetime:
    """Get the latest available rag from a specific source type: properties, examples"""
    try:
        latest_rag_date = get_latest_rag_date(source_type)
        
        rag_name = f"rag_{source_type}_{latest_rag_date.strftime('%Y-%m-%d_%H-%M-%S')}.zip"
        logger.info(f"Downloading most recent {source_type} RAG: {rag_name}")
        return download_rag(rag_name, chroma_dir_name, source_type)
            
    except Exception as e:
        logger.error(f"Error downloading most recent RAG for {source_type}: {str(e)}")
        raise


def download_vectorstore(source_type):
    persist_directory = get_vectorstore_directory(source_type)
    rag_date_file = os.path.join(persist_directory, "rag_date.txt")

    # Check existing RAG date
    if os.path.exists(rag_date_file):
        try:
            rag_date_content = open(rag_date_file, "r").read().strip()
            if rag_date_content:
                rag_date = dt.datetime.strptime(rag_date_content, "%Y-%m-%d")
            else:
                rag_date = None
        except ValueError as e:
            logger.warning(f"Invalid date format in {rag_date_file}: {e}")
            rag_date = None
    else:
        rag_date = None

    try:
       
        latest_rag = get_latest_rag_date(source_type)
        if rag_date is None or rag_date < latest_rag:
            logger.info(f"Downloading latest {source_type} RAG...")
            # Download the latest RAG
            rag_date = download_latest_rag(persist_directory, source_type)
            # create file with rag date
            with open(rag_date_file, "w") as f:
                f.write(rag_date.strftime("%Y-%m-%d"))
        return rag_date
    except Exception as e:
        logger.info(f"Failed to download latest RAG for {source_type}: {str(e)}")
        if os.path.exists(persist_directory):
            # Use existing vectorstore
            if not os.path.exists(rag_date_file):
                rag_date = dt.datetime.now()
                with open(rag_date_file, "w") as f:
                    f.write(rag_date.strftime("%Y-%m-%d"))
            else:
                rag_date = dt.datetime.strptime(open(rag_date_file, "r").read(), "%Y-%m-%d")
            return rag_date
        else:
            raise ValueError(f"No vectorstore available for {source_type} and download failed")


download_vectorstore("properties")