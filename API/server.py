from src import app

import torch
from flask import Flask, request

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)