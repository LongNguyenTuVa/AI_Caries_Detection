"""Perform test request"""
import pprint

import requests

import argparse
import io

import torch
from PIL import Image
from flask import Flask, request

DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"
TEST_IMAGE = "000739.jpg"

image_data = open(TEST_IMAGE, "rb").read()

response = requests.post(DETECTION_URL, files={"image": image_data}).json()

pprint.pprint(response)
