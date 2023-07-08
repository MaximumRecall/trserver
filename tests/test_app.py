import os
from uuid import uuid4

from cassandra.cluster import Cluster

from flaskr.db import DB
from flaskr.logic import save_if_new, recent_urls, search


db = DB(Cluster())
user_id = uuid4()

current_dir = os.path.dirname(os.path.abspath(__file__))

def test_all():
    with open(os.path.join(current_dir, 'resources', 'wsj.txt'), 'r') as file:
        content = file.read()
    assert save_if_new(db, 'localhost://wsj.txt', 'This is a test title longer than 15', content, str(user_id))
    assert not save_if_new(db, 'localhost://wsj.txt', 'This is a test title longer than 15', content, str(user_id))

    urls, oldest_saved_at = recent_urls(db, str(user_id))
    assert 'localhost://wsj.txt' in urls[0]['full_url']

    results = search(db, str(user_id), 'affirmative action')
    assert 'localhost://wsj.txt' in results[0]['full_url']


if __name__ == '__main__':
    test_all()
