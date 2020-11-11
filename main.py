from flask import jsonify
import flask
from flask import Flask
from flask_cors import CORS
import time

from detector import detect, test_news
from fakenews import fknws_two

app = Flask(__name__)
CORS(app)

#fknws_two()
device, model, tokenizer = detect()

@app.route('/api/fake/', methods=["POST"])
def fake_detector():
    json_data = flask.request.json
    news = json_data["news"]

    isFake, probability = test_news(news, device, model, tokenizer)

    return jsonify(
        isFake,
        probability
    )
    # return json.dumps({"isFake": True, "probability": 1})


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=6257)
