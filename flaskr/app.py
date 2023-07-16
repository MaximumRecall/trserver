import json
import os

from flask import Flask, request, render_template, jsonify, Response, stream_with_context

from . import logic
from .config import db
from .forms import SearchForm


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('WTF_SECRET')


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/search', methods=['GET'])
def search():
    user_id_str = request.args.get('user_id')
    if not user_id_str:
        return jsonify({"error": "user_id not provided"}), 400
    saved_before_str = request.args.get('saved_before')

    urls, oldest_saved_at = logic.recent_urls(db, user_id_str, saved_before_str)
    form = SearchForm(user_id_str=user_id_str)
    return render_template('search.html',
                           user_id_str=user_id_str,
                           saved_before_str=saved_before_str,
                           urls=urls,
                           oldest_saved_at=oldest_saved_at,
                           form=form)

@app.route('/results', methods=['POST'])
def results():
    form = SearchForm()
    if form.validate_on_submit():
        search_results = logic.search(db, form.user_id_str.data, form.search_text.data)
        return render_template('results.html', results=search_results, form=form)

    return jsonify({"error": "Invalid form data"}), 400


@app.route('/save_if_new', methods=['POST'])
def save_if_new():
    data = request.json

    url = data.get('url')
    title = data.get('title')
    text_content = data.get('text_content')
    user_id_str = data.get('user_id')

    if not url:
        return jsonify({"error": "url not provided"}), 400
    if not title:
        return jsonify({"error": "title not provided"}), 400
    if not text_content:
        return jsonify({"error": "text_content not provided"}), 400
    if not user_id_str:
        return jsonify({"error": "user_id not provided"}), 400

    result = logic.save_if_new(db, url, title, text_content, user_id_str)

    return jsonify({"saved": result}), 200


@app.route('/snapshot/<user_id_str>/<url_id_str>/', methods=['GET'])
@stream_with_context
def snapshot(user_id_str, url_id_str):
    title, formatted_content = logic.load_snapshot(db, user_id_str, url_id_str)
    return render_template('snapshot.html',
                           user_id_str=user_id_str,
                           url_id_str=url_id_str,
                           title=title,
                           formatted_content=formatted_content)

@app.route('/snapshot/stream/<user_id_str>/<path:url>/<saved_at_str>/', methods=['GET'])
@stream_with_context
def snapshot_stream(user_id_str, url, saved_at_str):
    def generate():
        for formatted_content in logic.load_snapshot(db, user_id_str, url, saved_at_str):
            yield 'data: {}\n\n'.format(json.dumps({"formatted_content": formatted_content}))
        yield 'event: EOF\ndata: {}\n\n'
    return Response(generate(), mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True)
