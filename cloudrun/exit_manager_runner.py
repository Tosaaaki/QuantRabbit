from cloudrun.exit_manager_service import app

if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)
