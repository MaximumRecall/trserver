import os

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from flask import Flask, request, jsonify

from pathlib import Path
import sys
cwd = str(Path(__file__).parent)
sys.path.append(cwd)
import logic

from db import DB
cloud_config= {
  'secure_connect_bundle': os.path.join(cwd, '<</PATH/TO/>>secure-connect-total-recall.zip')
}
auth_provider = PlainTextAuthProvider(os.environ.get('ASTRA_CLIENT_ID'), os.environ.get('ASTRA_CLIENT_SECRET'))
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
db = DB(cluster)

app = Flask(__name__)


@app.route("/")
def index():
    git_hash = os.environ.get('GIT_HASH', 'Git SHA not found')
    return f'Current Git SHA: {git_hash}'


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
