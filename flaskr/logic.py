import os
from datetime import datetime
from urllib.parse import urlparse
from uuid import UUID, uuid4

import nltk
import numpy as np
import openai
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from .db import DB
from .util import humanize_datetime

# nltk.download('punkt') # needed locally; in heroku this is done in nltk.txt
openai.api_key = os.environ.get('OPENAI_KEY')
if not openai.api_key:
    raise Exception('OPENAI_KEY environment variable not set')
tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
encoder = SentenceTransformer('multi-qa-MiniLM-L6-dot-v1')


def truncate_to(source, max_tokens):
    tokens = list(tokenizer.encode(source))
    truncated_tokens = []
    total_tokens = 0

    for token in tokens:
        total_tokens += 1
        if total_tokens > max_tokens:
            break
        truncated_tokens.append(token)

    truncated_s = tokenizer.decode(truncated_tokens)
    return truncated_s


summarize_prompt = ("You are a helpful assistant who will give the subject of the provided web page content in a single sentence. "
                    "Do not begin your response with 'The web page', or 'The subject is', just give the subject with no extra context. "
                    "Examples of good responses: "
                    "The significance of German immigrants in early Texas history, "
                    "The successes and shortcomings of persistent collections in server-side Java development.")
def summarize(text: str) -> str:
    truncated = truncate_to(text, 3900)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": summarize_prompt},
            {"role": "user", "content": truncated},
        ]
    )
    return response.choices[0].message.content


def _save_article(db: DB, path: str, text: str, url: str, title: str, user_id: uuid4) -> None:
    lines = [line for line in text.splitlines() if line]
    sentences = [nltk.sent_tokenize(line) for line in lines]  # list of lists of sentences
    flattened = [title] + [sentence for sublist in sentences for sentence in sublist]  # flatten
    vectors = encoder.encode(flattened, normalize_embeddings=True)
    db.upsert_chunks(user_id, path, url, title, text, zip(flattened, vectors))


def _is_different(text, last_version):
    """True if text is at least 5% different from last_version"""
    if last_version is None:
        return True

    vectorizer = CountVectorizer().fit_transform([text, last_version])
    vectors = vectorizer.toarray()
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    dot = np.dot(normalized[0], normalized[1])
    print("difference between this and previous version is " + str(dot))
    return dot < 0.95


def save_if_new(db: DB, url: str, title: str, text: str, user_id_str: str) -> bool:
    user_id = UUID(user_id_str)
    parsed = urlparse(url)
    path = parsed.hostname + parsed.path
    last_version = db.last_version(user_id, path)
    if not _is_different(text, last_version):
        return False

    if len(title) < 15:
        title = summarize(text)
    _save_article(db, path, text, url, title, user_id)
    return True


def recent_urls(db: DB, user_id_str: str, saved_before_str: str | None) -> tuple[list[dict[str, str | datetime]], datetime]:
    user_id = UUID(user_id_str)
    saved_before = datetime.fromisoformat(saved_before_str) if saved_before_str else None

    limit = 10
    results = db.recent_urls(user_id, saved_before, limit)
    oldest_saved_at = min(result['saved_at'] for result in results) if results and len(results) == limit else None
    for result in results:
        result['saved_at'] = humanize_datetime(result['saved_at'])
    return results, oldest_saved_at


def search(db: DB, user_id_str: str, search_text: str) -> list:
    vector = encoder.encode([search_text], normalize_embeddings=True)[0]
    results = db.search(UUID(user_id_str), vector)
    for result in results:
        result['saved_at'] = humanize_datetime(result['saved_at'])
    return results
