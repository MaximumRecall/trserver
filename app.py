from cassandra.cluster import Cluster
from flask import Flask, request, jsonify

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import logic

from db import DB


db = DB(Cluster())
app = Flask(__name__)

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
