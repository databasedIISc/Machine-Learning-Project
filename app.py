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

#Receiving the dataset here
@app.route("/upload", methods=['POST'])
def upload_dataset():
    global df
    if request.method == "POST":
        file = request.files["file"]
        if file:
            df = pd.read_csv(file)
            return render_template("upload1.html", message = "Dataset Uploaded Successfully")

        else:
            return render_template("upload.html", message = "Dataset Not Uploaded")


#Start
@app.route("/main_page")
def main_page():
    return render_template("main_page.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")