from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    #column names
    cols = list(df.columns)
    
    # Row 1:- no. of missing values
    misses = df.isnull().sum()
    
    # Row 2:- no. of present values
    present = list(df.count())
    
    # Row 3:- type of data
    dtypes = list(df.dtypes.astype("str"))
    
    # Row 4:- no. of unique values
    unique = list(df.nunique())
    
    row1 = pd.DataFrame([present], columns=cols)
    row2 = pd.DataFrame([misses], columns=cols)
    row3 = pd.DataFrame([dtypes], columns=cols)
    row4 = pd.DataFrame([unique], columns=cols)
   
    df_req = pd.concat([row1, row2, row3,row4], keys = ["Counts","Missing Values","Type","Unique Values"])
    df_req = df_req.droplevel(1)
    return render_template("insights1.html", dataset = df_req.to_html())

#insights2
@app.route("/insights2")
def insights2():
    global df_req
    cols = list(df.columns)
    
    num = df.select_dtypes(include=["int64","float64"])
    if num.empty:
        return render_template("insights1.html", dataset = "No Numerical Features Found");   
    
    # Row 1:- minimum values
    rep = "NC"
    min_val = list(df.min())
    min_val = [rep if type(x) == "str" else x for x in min_val]

    # Row 2:- maximum values
    max_val = list(df.max())
    max_val = [rep if type(x) == "str" else x for x in max_val]

    # Calculation for further steps
    df_copy = df.copy()
    array = []
    a = list(df.dtypes.astype("str"))
    c = list(df.columns)
    for i in range(len(a)):
        if a[i] == "object":
            change = c[i]
            array.append(i)
            df_copy[change] = -1

    # Row 3:- Difference in min and max values
    diff = df_copy.max() - df_copy.min()

    # Row 4:- Mean of the values
    mean = df_copy.mean()

    # Row 5:- Median of the values
    median = df_copy.median()

    # Row 6:- Mean/Median difference
    diff2 = list(np.array(mean) - np.array(median))

    # Row 7:- Standard Deviation of the values
    std = df_copy.std()

    row1 = pd.DataFrame([min_val], columns=cols)
    row2 = pd.DataFrame([max_val], columns=cols)
    row3 = pd.DataFrame([diff], columns=cols)
    row4 = pd.DataFrame([mean], columns=cols)
    row5 = pd.DataFrame([median], columns=cols)
    row6 = pd.DataFrame([diff2], columns=cols)
    row7 = pd.DataFrame([std], columns=cols)

    df_req = pd.concat([row1, row2, row3,row4, row5, row6, row7], keys = ["Minimum","Maximum","Difference","Mean","Median","Mean-Median Difference","Standard Deviation"])
    df_req = df_req.droplevel(1)

    for i in range(len(array)):
        index = array[i]
        change = c[index]
        df_req[change] = "NC"
        
        
    # this code will fail if all are categorical
    return render_template("insights2.html", dataset = df_req.to_html())

#round to 2 decimal
@app.route("/round2")
def round2():
    return render_template("insights2.html", dataset = df_req.round(2).to_html())

#round to 3 decimal
@app.route("/round3")
def round3():
    return render_template("insights2.html", dataset = df_req.round(3).to_html())

#Categorical Analysis
@app.route("/category")
def category():
    categorical_features = df.select_dtypes(include=["object"])
    if categorical_features.empty:
        return render_template("category.html", dataset = "No Categorical Features")
    return render_template("category.html", dataset = categorical_features.describe().to_html())

#Visualization-I
@app.route("/visual1")
def visual1():
    num = df.select_dtypes(include=["int64","float64"])
    if num.empty:
        return render_template("visualization1.html", message1 = "No Numerical Features Found")
    plt.figure(figsize=(10,10))
    sns.heatmap(num.corr().round(2), annot=True, cmap="coolwarm")
    plt.savefig("static/images/visual1/heatmap.png", bbox_inches = 'tight')
    plt.clf()
    sns.histplot(data=df,x=df.columns[-1], bins = 60, color = "r")
    plt.savefig("static/images/visual1/hist.png", bbox_inches = 'tight')
    
    return render_template("visualization1.html", graph1_url = "static/images/visual1/heatmap.png", message1 = "Correlation Heatmap", graph2_url = "static/images/visual1/hist.png", message2 = "Histogram of the Target Variable")

@app.route("/phase2")
def phase2():
    return render_template("missvalue.html")

    

if __name__=="__main__":
    app.run(host="0.0.0.0")
