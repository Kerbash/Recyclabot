import traceback
from datetime import datetime
import shutil

import requests
from flask import request, redirect, make_response, Blueprint, render_template, current_app
from werkzeug.utils import secure_filename
import os
import json

_dir = os.path.dirname(os.path.abspath(__file__))
upload_handler_blueprint = Blueprint('upload_handler_blueprint', __name__, template_folder="templates",
                                     static_folder="statics")


@upload_handler_blueprint.app_errorhandler(404)
def not_found(e):
    return "URL DOES NOT EXIST", 404


@upload_handler_blueprint.route('/')
def home():
    return render_template("main_page.html")


def storeData(client, item, amount):
    url = "http://api.ipstack.com/" + client.remote_addr + "? access_key = 7a627a706660413b061d75f609ede57b"
    r = requests.get(url)
    j = json.loads(r.text)
    try:
        continent_code = j['continent_code']
    except KeyError:
        continent_code = 'NA'

    map_data = current_app.config["MAP_DATA"]
    map_data[continent_code][item] += amount
    json.dump(map_data, open(current_app.config["MAP_DATA_JSON"], "w"))

    return (map_data[continent_code]["name"])


@upload_handler_blueprint.route('/uploading', methods=['POST'])
def upload_page():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(current_app.config["UPLOAD_FOLDER"] + filename)

    # check if the neural net has been initialized
    nn = current_app.config["NEURAL_NET"]
    result = nn.detect(current_app.config["UPLOAD_FOLDER"] + filename)
    result = nn.getMax(result)

    # interpret the result
    category = {
        0: "none recyclable",
        1: "compost",
        2: "cardboard",
        3: "metal",
        4: "paper",
        5: "plastic bottle"
    }
    string = f"|     1 {category[result].ljust(26, ' ')}|"
    continent = storeData(request, category[result], 1)
    continent = f"|   {continent.ljust(30, ' ')}|"

    return render_template("response.html", ITEM=string, LOCATION=continent)


@upload_handler_blueprint.route('/map')
def map_view():
    def compile(dic):
        output = ""
        for key in dic:
            output += key + ": " + str(dic[key]) + "\n"
        return output

    map_data = current_app.config["MAP_DATA"]

    NA = compile(map_data["NA"])
    SA = compile(map_data["SA"])
    AF = compile(map_data["AF"])
    OC = compile(map_data["OC"])
    AN = compile(map_data["AN"])
    AS = compile(map_data["AS"])
    EU = compile(map_data["EU"])

    print(NA)

    return render_template("map.html", NA=NA, SA=SA, AF=AF, OC=OC, AN=AN, AS=AS, EU=EU)

@upload_handler_blueprint.route('/incorrect')
def incorrect():
    # get age
    birth = current_app.config["BIRTH_TIME"]
    age = str(datetime.now() - birth)
    age_str = f"|   {age.ljust(30)}|"

    # TODO a way to feed the wrong data back into the neural network

    return render_template("incorrect.html", AGE = age_str)