from flask import Flask, request, jsonify
import logic


app = Flask(__name__)

@app.route('/is_article', methods=['POST'])
def is_article():
    data = request.json
    content = data.get('content')

    if not content:
        return jsonify({"error": "Content not provided"}), 400

    result = logic.is_article(content)

    return jsonify({"is_article": result}), 200


if __name__ == "__main__":
    app.run(debug=True)
