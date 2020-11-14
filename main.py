import flask
from flask import Flask
from flask import jsonify
from flask_cors import CORS

from detector import training_and_save_model
from fakenews import load_specific_model, detect_fake

app = Flask(__name__)
CORS(app)

#training_and_save_model()
device, model, tokenizer = load_specific_model()

@app.route('/api/fake/', methods=["POST"])
def fake_detector():
    json_data = flask.request.json
    news = json_data["news"]

    isFake, probability = detect_fake(news, device, model, tokenizer)
    return jsonify(
        isFake=isFake,
        probability=probability
    )
    # return jsonify(isFake: real, probability=1)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=6257)
