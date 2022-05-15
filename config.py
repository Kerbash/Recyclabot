import datetime
import json
import os
from flask import current_app
import pathlib

from NN.recognizer import Recognizer

class Config:
    UPLOAD_FOLDER = "temp\\"

    NEURAL_NET = Recognizer("C:\\Users\\guyok\\Desktop\\Coding\\Recyclabot\\NN\\model.pt")
    MAP_DATA_JSON = "data\\global_stats.json"
    MAP_DATA = json.load(open(MAP_DATA_JSON))

    BIRTH_TIME = datetime.datetime(2022, 5, 15, 12, 38, 58)

    USER_PHOTO_LOG = {}