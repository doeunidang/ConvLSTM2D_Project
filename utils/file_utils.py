# file_utils.py
import os

def ensure_directory_exists(directory):
    """디렉토리가 존재하지 않으면 생성."""
    if not os.path.exists(directory):
        os.makedirs(directory)
