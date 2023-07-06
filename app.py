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
            next_var=True
        else:
            next_var=False
    return render_template("upload.html", next_var=next_var)

#Start
@app.route("/main_page")
def main_page():
    return render_template("slot2intro.html")


#Show Dataset
@app.route("/show")
def show():
    return render_template("dataset.html", dataset = df.to_html())

#Insights1
@app.route("/insights1")
def insights1():
    # Code to be added
    return render_template("insights1.html")

#insights2
@app.route("/insights2")
def insights2():
    #code to be added
    return render_template("insights2.html")



if __name__=="__main__":
    app.run(host="0.0.0.0")