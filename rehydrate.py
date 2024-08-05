import os
import gzip
import json
from typing import Dict, Any

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
    for root, dirs, files in os.walk(tr_data_dir):
        for file in files:
            if file.endswith('.gz'):
                file_path = os.path.join(root, file)
                
                # Skip if already processed
                if is_processed(file_path):
                    print(f"Skipping already processed file: {file_path}")
                    continue

                user_id = os.path.basename(os.path.dirname(file_path))

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

                # save to db
                save_if_new(db, url, title, text_content, user_id_str)
                
                # Mark as processed
                mark_as_processed(file_path)
                
                print(f"Reloaded: {file_path}")


if __name__ == "__main__":
    rehydrate()
