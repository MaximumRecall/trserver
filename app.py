import os

from cassandra.cluster import Cluster
from flask import Flask, request, render_template, jsonify

from .db import DB
from . import logic
from .forms import SearchForm

from db import DB
cloud_config = {
  'secure_connect_bundle': os.path.join(cwd, 'secure-connect-total-recall.zip')
}
astra_client_id = os.environ.get('ASTRA_CLIENT_ID')
if not astra_client_id:
    raise Exception('ASTRA_CLIENT_ID environment variable not set')
astra_client_secret = os.environ.get('ASTRA_CLIENT_SECRET')
if not astra_client_secret:
    raise Exception('ASTRA_CLIENT_SECRET environment variable not set')
auth_provider = PlainTextAuthProvider(astra_client_id, astra_client_secret)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
db = DB(cluster)
# running locally
# db = DB(Cluster())

app = Flask(__name__)


@app.route("/")
def index():
    git_hash = os.environ.get('GIT_HASH', 'Git SHA not found')
    return f"It's alive! Current Git SHA: {git_hash}"


app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(32)


@app.route('/search', methods=['GET'])
def search():
    user_id_str = request.args.get('user_id_str')
    if not user_id_str:
        return jsonify({"error": "user_id not provided"}), 400

    urls = logic.recent_urls(db, user_id_str)
    form = SearchForm(user_id_str=user_id_str)
    return render_template('search.html', urls=urls, form=form)


@app.route('/results', methods=['POST'])
def results():
    form = SearchForm()
    if form.validate_on_submit():
        search_results = logic.search(db, form.user_id_str.data, form.search_text.data)
        return render_template('results.html', results=search_results)

    return jsonify({"error": "Invalid form data"}), 400



@app.route('/save_if_article', methods=['POST'])
def save_if_article():
    data = request.json

    html_content = data.get('html_content')
    url = data.get('url')
    user_id_str = data.get('user_id')

    if not html_content:
        return jsonify({"error": "html_content not provided"}), 400
    if not url:
        return jsonify({"error": "url not provided"}), 400
    if not user_id_str:
        return jsonify({"error": "user_id not provided"}), 400

    result = logic.save_if_article(db, html_content, url, user_id_str)

    return jsonify({"saved": result}), 200


if __name__ == "__main__":
    app.run(debug=True)
