from flask import Flask, request

app = Flask(__name__)


@app.route("/")
def health():
    return "OK", 200


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)
