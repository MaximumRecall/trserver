import os
from datetime import datetime
from urllib.parse import urlparse
from uuid import uuid1, UUID, uuid4

import nltk
import openai
import tiktoken
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

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


article_prompt = ("You are a helpful assistant who will determine if the provided web page content "
                  "is an article consisting mostly of text, or something else. "
                  "Respond with Article, Other, or Unsure.")
def _is_article(text: str) -> bool:
    truncated = truncate_to(text, 4000)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": article_prompt},
            {"role": "user", "content": truncated},
        ]
    )

    def answer(r):
        return r.choices[0].message.content.lower()

    if answer(response) == "unsure":
        if len(truncated) < 0.6 * len(text):
            # if we had to truncate the content significantly, try again with more context
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": article_prompt},
                    {"role": "user", "content": truncate_to(text, 15900)},
                ]
            )
        else:
            # try again with more capable model
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": article_prompt},
                    {"role": "user", "content": text},
                ]
            )

    return answer(response) == "article"


def _save_article(db: DB, text: str, url: str, title: str, user_id: uuid4) -> None:
    sentences = nltk.sent_tokenize(text)
    vectors = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [(uuid1(), sentence, v) for sentence, v in zip(sentences, vectors)]
    db.upsert_chunks(user_id, url, title, chunks)


def save_if_article(db: DB, html_content: str, url: str, user_id_str: str) -> bool:
    user_id = UUID(user_id_str)
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(" ", strip=True)
    if not _is_article(text):
        return False

    title = soup.title.string.strip() if soup.title else ""
    if len(title) < 15:
        title = summarize(text)
    _save_article(db, text, url, title, user_id)
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

# for testing
def is_article(html_content: str, url: str = None) -> bool:
    # if url starts with reddit.com/r/.*/comments/ then return true
    if url:
        parsed_url = urlparse(url)
        # reddit comments threads are treated as articles
        if ((parsed_url.netloc.startswith('www.reddit.com') or parsed_url.netloc.startswith('reddit.com'))
                and parsed_url.path.startswith('/r/') and '/comments/' in parsed_url.path):
            return True
    text = BeautifulSoup(html_content, 'html.parser').get_text(" ", strip=True)
    return _is_article(text)

def save_article(db: DB, html_content: str, url: str, user_id: uuid4) -> None:
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(" ", strip=True)
    title = soup.title.string.strip()
    _save_article(db, text, url, title, user_id)
