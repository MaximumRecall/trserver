import os
import sys
from pathlib import Path
from uuid import uuid4

from cassandra.cluster import Cluster

current_dir = Path(__file__).parent
sys.path.append(os.path.join(current_dir, '..'))
from logic import is_article, save_article
from db import DB


db = DB(Cluster())
user_id = uuid4()

def test_article():
    with open(os.path.join(current_dir, 'resources', 'article.html'), 'r') as file:
        content = file.read()
    assert is_article(content)

def test_not_article():
    with open(os.path.join(current_dir, 'resources', 'not-article.html'), 'r') as file:
        content = file.read()
    assert not is_article(content)

def test_save_article():
    with open(os.path.join(current_dir, 'resources', 'article.html'), 'r') as file:
        content = file.read()
    save_article(db, content, 'http://example.com', 'Example', user_id)


if __name__ == '__main__':
    test_save_article()
