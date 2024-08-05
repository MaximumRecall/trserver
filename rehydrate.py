import os
import gzip
import json
from typing import Dict, Any
from uuid import uuid1

from flaskr.config import db, tr_data_dir
from flaskr.logic import save_if_new, _is_different

def is_processed(file_path: str) -> bool:
    """Check if a marker file exists for the given file path."""
    marker_path = f"{file_path}.processed"
    return os.path.exists(marker_path)

def mark_as_processed(file_path: str) -> None:
    """Create a marker file for the given file path."""
    marker_path = f"{file_path}.processed"
    open(marker_path, 'w').close()

def rehydrate():
    # Walk through the tr_data_dir
    all_files = []
    for root, dirs, files in os.walk(tr_data_dir):
        for file in files:
            if file.endswith('.gz'):
                file_path = os.path.join(root, file)
                if is_processed(file_path):
                    print(f"Skipping already processed file: {file_path}")
                all_files.append(file_path)

    # Sort files based on timestamps
    sorted_files = sorted(all_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # Process sorted files
    for file_path in sorted_files:
        user_id = os.path.basename(os.path.dirname(file_path))

        # Parse timestamp from filename and create UUID1
        timestamp_ns = int(os.path.splitext(os.path.basename(file_path))[0])
        url_id = uuid1(clock_seq=timestamp_ns)

        # Read and parse the gzipped JSON file
        with gzip.open(file_path, 'rt') as f:
            data: Dict[str, Any] = json.load(f)

        # Extract necessary information
        url = data['url']
        title = data['title']
        text_content = data['text_content']
        user_id_str = data['user_id']

        # Ensure the user_id in the filename matches the one in the JSON
        assert user_id == user_id_str, f"User ID mismatch in {file_path}"
        # Save to db
        save_if_new(db, url, title, text_content, user_id_str, url_id)

        # Mark as processed
        mark_as_processed(file_path)
        print(f"Reloaded: {file_path}")


if __name__ == "__main__":
    rehydrate()
