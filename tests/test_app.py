import os
from uuid import uuid4

from cassandra.cluster import Cluster

from db import DB
from logic import is_article, save_article, summarize

db = DB(Cluster())
user_id = uuid4()

def test_article():
    with open(os.path.join(current_dir, 'resources', 'article.html'), 'r') as file:
        content = file.read()
    assert is_article(content)

def test_reddit_article():
    assert is_article("", "https://www.reddit.com/r/Xreal/comments/13xgvnv/what_is_3dof_screen_mirroring_and_what_does_xreal/")

def test_not_article():
    with open(os.path.join(current_dir, 'resources', 'not-article.html'), 'r') as file:
        content = file.read()
    assert not is_article(content)

def test_save_article():
    with open(os.path.join(current_dir, 'resources', 'article.html'), 'r') as file:
        content = file.read()
    save_article(db, content, 'http://example.com', user_id)

def test_summarize():
    with open(os.path.join(current_dir, 'resources', 'article.html'), 'r') as file:
        content = file.read()
    print(summarize(content))


if __name__ == '__main__':
    test_reddit_article()
