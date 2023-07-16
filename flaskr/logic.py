import os
from datetime import datetime, timedelta
from typing import Optional, List
from urllib.parse import urlparse
from uuid import UUID, uuid4

import nltk
import numpy as np
import openai
from sklearn.feature_extraction.text import CountVectorizer
import tiktoken
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from .db import DB
from .util import humanize_datetime


nltk.download('punkt') # needed locally; in heroku this is done in nltk.txt

openai.api_key = os.environ.get('OPENAI_KEY')
if not openai.api_key:
    raise Exception('OPENAI_KEY environment variable not set')
_tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')


class E5Encoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(self, inputs: List[str], normalize_embeddings=True) -> List[List[float]]:
        batch_dict = self.tokenizer(inputs, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        if normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Convert tensor to list of lists
        return embeddings.tolist()
_encoder = E5Encoder()


def truncate_to(source, max_tokens):
    tokens = list(_tokenizer.encode(source))
    truncated_tokens = []
    total_tokens = 0

    for token in tokens:
        total_tokens += 1
        if total_tokens > max_tokens:
            break
        truncated_tokens.append(token)

    truncated_s = _tokenizer.decode(truncated_tokens)
    return truncated_s


_summarize_prompt = ("You are a helpful assistant who will give the subject of the provided web page content in a single sentence. "
                     "Do not begin your response with any prefix."
                     "Give the subject in a form appropriate for an article or book title with no extra preamble or context."
                     "Examples of good responses: "
                     "The significance of German immigrants in early Texas history, "
                     "The successes and shortcomings of persistent collections in server-side Java development, "
                     "A personal account of the benefits of intermittent fasting.")
def summarize(text: str) -> str:
    truncated = truncate_to(text, 3900)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": _summarize_prompt},
            {"role": "user", "content": truncated},
        ]
    )
    return response.choices[0].message.content


def _save_article(db: DB, path: str, text: str, url: str, title: str, user_id: uuid4) -> None:
    # require that sentences contain an alphabetical character
    # (so that it doesn't match lines that are just a bunch of punctuation)
    sentences = [nltk.sent_tokenize(text)]
    flattened = [title] + [sentence.strip() for sentence in sentences
                           if any(c.isalpha() for c in sentence)]
    vectors = _encoder.encode(flattened, normalize_embeddings=True)
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


_format_prompt = ("You are a helpful assistant who will reformat raw text as html. "
                  "Add paragraphing and headings where appropriate. "
                  "Use bootstrap CSS classes.")
def _group_sentences_by_tokens(sentences, max_tokens):
    grouped_sentences = []
    current_group = []
    current_token_count = 0

    # Group sentences in chunks of max_tokens
    for sentence in sentences:
        token_count = len(list(_tokenizer.encode(sentence)))
        if current_token_count + token_count <= max_tokens:
            current_group.append(sentence)
            current_token_count += token_count
        else:
            grouped_sentences.append(current_group)
            current_group = [sentence]
            current_token_count = token_count

    # Add the last group if it's not empty
    if current_group:
        grouped_sentences.append(current_group)

    return grouped_sentences
def _ai_format(text_content):
    sentences = [sentence.strip() for sentence in nltk.sent_tokenize(text_content)]
    sentence_groups = _group_sentences_by_tokens(sentences, 6000)

    for group in sentence_groups:
        group_text = ' '.join(group)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": _format_prompt},
                {"role": "user", "content": group_text},
            ],
            stream=True
        )
        for response_piece in response:
            if response_piece and 'content' in response_piece['choices'][0]['delta']:
                yield response_piece['choices'][0]['delta']['content']


def _uuid1_to_datetime(uuid1):
    # UUID timestamps are in 100-nanosecond units since 15th October 1582
    uuid_datetime = datetime(1582, 10, 15) + timedelta(microseconds=uuid1.time // 10)
    return uuid_datetime


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


def recent_urls(db: DB, user_id_str: str, saved_before_str: Optional[str] = None) -> tuple[list[dict[str, Optional[str]]], datetime]:
    user_id = UUID(user_id_str)
    saved_before = datetime.fromisoformat(saved_before_str) if saved_before_str else None

    limit = 10
    results = db.recent_urls(user_id, saved_before, limit)
    for result in results:
        result['saved_at'] = _uuid1_to_datetime(result['url_id'])
        result['saved_at_human'] = humanize_datetime(result['saved_at'])
    oldest_saved_at = min(result['saved_at'] for result in results) if results and len(results) == limit else None
    print('saved urls are ' + str(results))
    return results, oldest_saved_at


def search(db: DB, user_id_str: str, search_text: str) -> list:
    vector = _encoder.encode([search_text], normalize_embeddings=True)[0]
    results = db.search(UUID(user_id_str), vector)
    for result in results:
        dt = _uuid1_to_datetime(result['url_id'])
        result['saved_at_human'] = humanize_datetime(dt)
    return results


def load_snapshot(db: DB, user_id_str: str, url_id_str: str) -> tuple[str, str]:
    user_id = UUID(user_id_str)
    url_id = UUID(url_id_str)
    _, title, _, formatted_content = db.load_snapshot(user_id, url_id)
    return title, formatted_content

def stream_snapshot(db: DB, user_id_str: str, url_id_str: str) -> tuple[str, str]:
    user_id = UUID(user_id_str)
    url_id = UUID(url_id_str)
    url_id, title, text_content, formatted_content = db.load_snapshot(user_id, url_id)

    formatted_pieces = []
    for piece in _ai_format(text_content):
        formatted_pieces.append(piece)
        yield piece
    formatted_content = ' '.join(formatted_pieces)
    db.save_formatting(user_id, url_id, path, formatted_content)
