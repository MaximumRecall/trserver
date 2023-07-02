from bs4 import BeautifulSoup
import openai
import tiktoken


openai.api_key = open('openai.key', 'r').read().splitlines()[0]

tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

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


prompt = ("You are a helpful assistant who will determine if the provided web page content "
          "is an article consisting mostly of text, or something else. "
          "Respond with Article, Other, or Unsure.")

def is_article(html_content: str) -> bool:
    content = BeautifulSoup(html_content, 'html.parser').get_text()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": truncate_to(content, 4000)},
        ]
    )

    def answer(r):
        return r.choices[0].message.content.lower()

    if answer(response) == "unsure":
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": truncate_to(content, 15900)},
            ]
        )

    return answer(response) == "article"
