import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(os.path.join(current_dir, '..'))
from logic import is_article


def test_article():
    with open(os.path.join(current_dir, 'resources', 'article.html'), 'r') as file:
        content = file.read()
    assert is_article(content)

def test_not_article():
    with open(os.path.join(current_dir, 'resources', 'not-article.html'), 'r') as file:
        content = file.read()
    assert not is_article(content)
