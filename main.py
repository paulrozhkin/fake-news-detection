import flask
from flask import Flask
from flask import jsonify
from flask_cors import CORS
from fakenews import load_specific_model, detect_fake


def create_app():
    app = Flask(__name__)
    CORS(app)

    device, model, tokenizer = load_specific_model()

    # Uncomment for re-train
    # training_and_save_model()

    @app.route('/api/fake/', methods=["POST"])
    def fake_detector():
        json_data = flask.request.json
        news = json_data["news"]

        is_real, probability = detect_fake(news, device, model, tokenizer)
        return jsonify(
            isReal=is_real,
            probability=probability
        )

    return app


if __name__ == "__main__":
    create_app().run(host='127.0.0.1', port=6257)
