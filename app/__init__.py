# from .DataBase import db as System_FileDB
import os

from .routes.chat_api import bp as chats_bp


def create_app(app):
    app.register_blueprint(chats_bp)
    app.secret_key = os.urandom(24)
    return app