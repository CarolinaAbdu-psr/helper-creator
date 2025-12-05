import os
import logging
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get AWS credentials and configuration from environment variables
AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION: Optional[str] = os.getenv("AWS_REGION")
S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME")

# Validate essential environment variables
if not S3_BUCKET_NAME:
    logger.error("S3_BUCKET_NAME environment variable is not set. This is required.")
    raise ValueError("S3_BUCKET_NAME environment variable is not set.")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
    logger.warning("One or more AWS credential/region environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION) are not set. The SDK will attempt to use its default credential chain.")

# Create an S3 client
s3_client = None
try:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    # You could add a test call here if needed, e.g., s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
    # to ensure connectivity and bucket existence early. This requires s3:ListBucket permission on the bucket.
    logger.info("S3 client created successfully.")
except Exception as e:
    logger.error(f"Failed to create S3 client: {e}")
    # Depending on the application, you might want to exit or raise the exception.
    # For now, functions will check if s3_client is None.

def upload_file_to_s3(file_path: str, object_name: Optional[str] = None) -> bool:
    """Upload a file to the configured S3 bucket.

    :param file_path: Path to the file to upload.
    :param object_name: S3 object name. If not specified, the base name of file_path is used.
    :return: True if file was uploaded successfully, False otherwise.
    """
    if not s3_client:
        logger.error("S3 client is not initialized. Cannot upload file.")
        return False
    if not S3_BUCKET_NAME: # Should be caught by initial check, but good for safety
        logger.error("S3_BUCKET_NAME is not configured. Cannot upload file.")
        return False
        
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    if object_name is None:
        object_name = os.path.basename(file_path)

    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, object_name)
        logger.info(f"Successfully uploaded {file_path} to {S3_BUCKET_NAME}/{object_name}")
        return True
    except ClientError as e:
        logger.error(f"Error uploading file {file_path} to {S3_BUCKET_NAME}/{object_name}: {e}")
        return False
    except FileNotFoundError: # Should be caught by os.path.exists, but defensive
        logger.error(f"File not found during s3_client.upload_file: {file_path}")
        return False
    except Exception as e: # Catch-all for other unexpected errors
        logger.error(f"An unexpected error occurred during upload of {file_path}: {e}")
        return False

def download_file_from_s3(object_name: str, file_path: Optional[str] = None) -> bool:
    """Download a file from the configured S3 bucket.

    :param object_name: S3 object name (key).
    :param file_path: Local file path to save the downloaded file.
                      If not specified, object_name is used as the local file name in the current directory.
    :return: True if file was downloaded successfully, False otherwise.
    """
    if not s3_client:
        logger.error("S3 client is not initialized. Cannot download file.")
        return False
    if not S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME is not configured. Cannot download file.")
        return False

    if file_path is None:
        file_path = object_name
    
    # Ensure directory for the local file path exists
    local_dir = os.path.dirname(file_path)
    if local_dir and not os.path.exists(local_dir):
        try:
            os.makedirs(local_dir)
            logger.info(f"Created directory {local_dir} for downloaded file.")
        except OSError as e:
            logger.error(f"Could not create directory {local_dir}: {e}")
            return False

    try:
        s3_client.download_file(S3_BUCKET_NAME, object_name, file_path)
        logger.info(f"Successfully downloaded {object_name} from {S3_BUCKET_NAME} to {file_path}")
        return True
    except ClientError as e:
        logger.error(f"Error downloading {object_name} from {S3_BUCKET_NAME} to {file_path}: {e}")
        # More specific error check, e.g., for 'NoSuchKey' or '404'
        if e.response.get('Error', {}).get('Code') == '404':
            logger.error(f"Object {object_name} not found in bucket {S3_BUCKET_NAME}.")
        return False
    except Exception as e: # Catch-all for other unexpected errors
        logger.error(f"An unexpected error occurred during download of {object_name}: {e}")
        return False

def list_files_in_s3() -> List[str]:
    """List files (object keys) in the configured S3 bucket.

    Handles pagination to list all objects if there are more than 1000.

    :return: A list of object keys. Returns an empty list if the bucket is empty or an error occurs.
    """
    if not s3_client:
        logger.error("S3 client is not initialized. Cannot list files.")
        return []
    if not S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME is not configured. Cannot list files.")
        return []

    object_keys: List[str] = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=S3_BUCKET_NAME)

        for page in page_iterator:
            if "Contents" in page:
                for item in page["Contents"]:
                    object_keys.append(item['Key'])
        
        if object_keys:
            logger.info(f"Found {len(object_keys)} file(s) in bucket {S3_BUCKET_NAME}.")
            # Optionally log all keys, can be verbose for many files:
            # for key in object_keys:
            #     logger.debug(f"- {key}")
        else:
            logger.info(f"Bucket {S3_BUCKET_NAME} is empty or no files found.")
        return object_keys
    except ClientError as e:
        logger.error(f"Error listing files in bucket {S3_BUCKET_NAME}: {e}")
        return []
    except Exception as e: # Catch-all for other unexpected errors
        logger.error(f"An unexpected error occurred while listing files from {S3_BUCKET_NAME}: {e}")
        return []

# Example Usage:
if __name__ == "__main__":
    # This block demonstrates how to use the functions.
    # Ensure your .env file is correctly set up with:
    # AWS_ACCESS_KEY_ID=your_access_key
    # AWS_SECRET_ACCESS_KEY=your_secret_key
    # AWS_REGION=your_region
    # S3_BUCKET_NAME=your_bucket_name

    if not s3_client or not S3_BUCKET_NAME:
        logger.error("S3 client or bucket name not configured. Aborting example usage.")
    else:
        logger.info(f"--- Starting S3 API Demo for bucket: {S3_BUCKET_NAME} ---")

        # 1. Create a dummy file for upload
        sample_file_to_upload = "my_sample_file.txt"
        s3_object_key = f"{sample_file_to_upload}" # Store in a "folder"

        with open(sample_file_to_upload, "w") as f:
            f.write("Hello from the S3 API script!\nThis is a test.")
        logger.info(f"Created dummy file: {sample_file_to_upload}")

        # 2. Upload the file
        logger.info(f"\n--- Attempting to upload: {sample_file_to_upload} to {s3_object_key} ---")
        if upload_file_to_s3(sample_file_to_upload, s3_object_key):
            logger.info("Upload successful.")
        else:
            logger.error("Upload failed.")

        # 3. List files in the bucket
        logger.info(f"\n--- Listing files in bucket: {S3_BUCKET_NAME} ---")
        files_in_bucket = list_files_in_s3()
        if files_in_bucket:
            logger.info("Files currently in bucket:")
            for file_key in files_in_bucket:
                logger.info(f"  - {file_key}")
        else:
            logger.info("No files found or an error occurred during listing.")

        # 4. Download the uploaded file (if it was uploaded and listed)
        downloaded_file_path = f"downloaded_{sample_file_to_upload}"
        if s3_object_key in files_in_bucket:
            logger.info(f"\n--- Attempting to download: {s3_object_key} to {downloaded_file_path} ---")
            if download_file_from_s3(s3_object_key, downloaded_file_path):
                logger.info("Download successful.")
                # Verify content (optional)
                try:
                    with open(downloaded_file_path, "r") as f_read:
                        content = f_read.read()
                        logger.info(f"Content of downloaded file '{downloaded_file_path}':\n{content}")
                except IOError as e:
                    logger.error(f"Could not read downloaded file: {e}")
                # Clean up downloaded file
                os.remove(downloaded_file_path)
                logger.info(f"Cleaned up downloaded file: {downloaded_file_path}")
            else:
                logger.error("Download failed.")
        else:
            logger.warning(f"Skipping download test as '{s3_object_key}' was not found in the bucket listing (it might have failed to upload).")

        # 5. Attempt to download a file
        logger.info("\n--- Attempting to download a non-existent file ---")
        download_file_from_s3("my_sample_file.txt", "my_sample_file_downloaded.txt")

        # Clean up the local dummy file
        if os.path.exists(sample_file_to_upload):
            os.remove(sample_file_to_upload)
            logger.info(f"Cleaned up local dummy file: {sample_file_to_upload}")
        
        logger.info("\n--- S3 API Demo Finished ---")
