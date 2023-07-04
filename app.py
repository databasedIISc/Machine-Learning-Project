from flask import Flask, render_template, request
import pandas as pd
import numpy as np

# This is our main python file that will run the flask app

app = Flask(__name__)

#Home Page
@app.route("/")
def home():
    return render_template("index.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")