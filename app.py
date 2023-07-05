import os

from flask import Flask, request, render_template, jsonify

from . import logic
from .config import db
from .forms import SearchForm


app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(32)


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


if __name__ == "__main__":
    app.run(debug=True)
