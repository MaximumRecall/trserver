import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(os.path.join(current_dir, '..'))
from logic import save_if_article


def test_article():
    with open(os.path.join(current_dir, 'resources', 'article.html'), 'r') as file:
        content = file.read()
    assert save_if_article(content)

def test_not_article():
    with open(os.path.join(current_dir, 'resources', 'not-article.html'), 'r') as file:
        content = file.read()
    assert not save_if_article(content)


if __name__ == '__main__':
    test_article()
    test_not_article()
