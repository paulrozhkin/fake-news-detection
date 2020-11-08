from flask import jsonify
import flask
from flask import Flask
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)


@app.route('/api/fake/', methods=["POST"])
def fake_detector():
    json_data = flask.request.json
    news = json_data["news"]
    time.sleep(1)

    return jsonify(
        isFake=True,
        probability=1
    )
    # return json.dumps({"isFake": True, "probability": 1})


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=6257)
