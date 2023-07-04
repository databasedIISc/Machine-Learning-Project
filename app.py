from flask import Flask, render_template, request
import pandas as pd
import numpy as np

# This is our main python file that will run the flask app

app = Flask(__name__)

#Home Page
@app.route("/")
def home():
    return render_template("index.html")

#Introduction Page
@app.route("/introduction")
def introduction():
    return render_template("intro.html")

#Upload Page
@app.route("/upload")
def upload():
    return render_template("upload.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")