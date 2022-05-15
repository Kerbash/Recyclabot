from flask import Flask, request, current_app
from threading import Thread

import datetime
import json
import os


from NN.recognizer import Recognizer
from upload.blueprint import upload_handler_blueprint
from config import Config

class Recyclabot(Thread):
    def __init__(self, name="Recyclabot"):
        super().__init__()
        self.webApp = Flask(name)

        # load the blueprints
        self.webApp.register_blueprint(upload_handler_blueprint)

        # load the config
        self.webApp.config.from_object(Config)

    def run(self):
        self.webApp.run(host="10.0.0.206", port="8080")

if __name__ == "__main__":
    Recyclabot().run()

    # training first


"""    from NN.recognizer import Recognizer
    r = Recognizer("C:\\Users\\guyok\\Desktop\\Coding\\Recyclabot\\NN\\model.pt")
    for image in os.listdir("NN\\train\\cardboard"):
        if image[-3:] == "png":
            d = r.detect("NN\\train\\cardboard\\" + image)
            print(r.getMax(d))"""
