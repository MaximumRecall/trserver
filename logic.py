import os
from uuid import uuid1, UUID, uuid4

from bs4 import BeautifulSoup
import openai
import tiktoken
from sentence_transformers import SentenceTransformer
import nltk
# nltk.download('punkt')

from db import DB
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
    db.upsert_batch(user_id, url, title, chunks)


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


# for testing
def is_article(html_content: str) -> bool:
    text = BeautifulSoup(html_content, 'html.parser').get_text(" ", strip=True)
    return _is_article(text)

def save_article(db: DB, html_content: str, url: str, user_id: uuid4) -> None:
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(" ", strip=True)
    title = soup.title.string.strip()
    _save_article(db, text, url, title, user_id)
