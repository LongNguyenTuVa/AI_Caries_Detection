from flask import abort, Flask, jsonify, request
from flask_cors import CORS
 
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)

from src.controllers.Predict import Predict