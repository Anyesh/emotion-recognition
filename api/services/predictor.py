
from emotion_detection.config import config
from emotion_detection.utils import file_utils
from emotion_detection.main import test_pipeline
from flask import request, render_template
import flask
import time
import functools
import joblib
import json
from ..settings import shared_components
from ..utils import response_interface
import datetime

memory = joblib.Memory("../data/", verbose=0)



def home():
    return render_template("index.html")

def logs():
    db = shared_components["db"]
    collection = db.outputs
    data = collection.find({}).sort([('timestamp', -1)]).limit(10)

    return render_template("logs.html", data=list(data))


# @memory.cache
def sentence_prediction(model_name, sentence):
    # text_processor = load_text_processor()
    # label_encoder = load_label_processor()
    output, proab, exp = test_pipeline(model_name=model_name, input_text=sentence)

    return output[0], proab, exp


def predict():
    payload = request.get_json()
    start_time = time.time()

    try:
        prediction, probabilities, explainer = sentence_prediction(payload["model_name"], payload["text"])
        
        ## DB
        db = shared_components["db"]
        collection = db.outputs
      

        data = {
            "model_name": payload["model_name"], 
            "text": payload["text"],
            "prediction": str(prediction),
            "sentiment": "P" if  str(prediction) == 'joy' else "N",
            "probabilities": str(probabilities),
            "time_taken":  str(time.time() - start_time),
            "timestamp": datetime.datetime.utcnow()
        }

        res = response_interface(data=data, msg= "Prediction ran successfully.", status=200, explainer=explainer)

       
        collection.insert_one(dict(data))
        return flask.jsonify(res)
    except Exception as e:
        res = response_interface(data={},msg="There was an error during prediction.", status=400 )
     
        return flask.jsonify(res), 400


