import os
import gzip
import json
from uuid import UUID
from typing import Dict, Any

from config import db, tr_data_dir
from logic import save_if_new, _is_different


def rehydrate():
    # Walk through the tr_data_dir
    for root, dirs, files in os.walk(tr_data_dir):
        for file in files:
            if file.endswith('.gz'):
                file_path = os.path.join(root, file)
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
                print(f"Reloaded: {file_path}")


if __name__ == "__main__":
    rehydrate()