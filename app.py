from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import io
import base64
matplotlib.use('Agg')
plt=matplotlib.pyplot

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
   
    df_req = pd.concat([row1, row2, row3,row4], keys = ["No. of Values","Missing Values","Data Type","Unique Values"])
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
    min_val = [rep if type(x) == str else x for x in min_val]

    # Row 2:- maximum values
    max_val = list(df.max())
    max_val = [rep if type(x) == str else x for x in max_val]

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
    
    arr = categorical_features.columns.tolist()
    return render_template("category.html", dataset = categorical_features.describe().to_html(), columns = arr)

#Visualization-I
@app.route("/visual1", methods = ["GET", "POST"])
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
    
    
    # for histograms
    columns = list(num.columns)    
    return render_template("visualization1.html", graph1_url = "static/images/visual1/heatmap.png", message1 = "Correlation Heatmap", graph2_url = "static/images/visual1/hist.png", message2 = "Histogram of the Target Variable",columns=columns)


@app.route("/histograms", methods = ["GET", "POST"])
def histograms():
    arr = request.form.getlist('columns')
    arr = [i.replace(","," ") for i in arr]
    
    if(len(arr) == 1):
        return render_template("visualization2.html", message = "Select at least 2 features")
    if(len(arr) > 9):
        return render_template("visualization2.html", message = "Select maximum 9 features")
    if len(arr) != 0:
        sns.set_style("darkgrid")
        
        if(len(arr) == 1):
            sns.histplot(x = df[arr[0]], bins = 30, kde = True, color = "r")
        
        elif(len(arr) > 1 and len(arr) < 5):
            fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15,15))
            for i in range(len(arr)):
                if (i == 0):
                    sns.histplot(x = df[arr[0]], ax = axes[0, 0], bins = 30, kde = True, color = "r")
                if (i == 1):
                    sns.histplot(x = df[arr[1]], ax = axes[0, 1], bins = 30, kde = True, color = "b")
                if (i == 2):
                    sns.histplot(x = df[arr[2]], ax = axes[1, 0], bins = 30, kde = True, color = "b")
                if (i == 3):
                    sns.histplot(x = df[arr[3]], ax = axes[1, 1], bins = 30, kde = True, color = "black")
                    
        else:
            fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (15,15))
            for i in range(len(arr)):
                if (i == 0):
                    sns.histplot(x = df[arr[0]], ax = axes[0, 0], bins = 30, kde = True, color = "r")
                if (i == 1):
                    sns.histplot(x = df[arr[1]], ax = axes[0, 1], bins = 30, kde = True, color = "b")
                if (i == 2):
                    sns.histplot(x = df[arr[2]], ax = axes[0, 2], bins = 30, kde = True, color = "black")
                if (i == 3):
                    sns.histplot(x = df[arr[3]], ax = axes[1, 0], bins = 30, kde = True, color = "b")
                if (i == 4):
                    sns.histplot(x = df[arr[4]], ax = axes[1, 1], bins = 30, kde = True, color = "r")
                if (i == 5):
                    sns.histplot(x = df[arr[5]], ax = axes[1, 2], bins = 30, kde = True, color = "g")
                if (i == 6):
                    sns.histplot(x = df[arr[6]], ax = axes[2, 0], bins = 30, kde = True, color = "black")
                if (i == 7):
                    sns.histplot(x = df[arr[7]], ax = axes[2, 1], bins = 30, kde = True, color = "g")
                if (i == 8):
                    sns.histplot(x = df[arr[8]], ax = axes[2, 2], bins = 30, kde = True, color = "r")
                    
        plt.savefig("static/images/visual1.1/histograms.png", bbox_inches = 'tight') 
        return render_template("visualization2.html", graph3_url = "static/images/visual1.1/histograms.png", message = "Histograms of the Selected Features")
    else:
        return render_template("visualization2.html", message = "Please select atleast one feature")
        

@app.route("/phase2")
def phase2():
    global null_df_copy
    null_df = pd.DataFrame(df.isnull().sum().astype(int), columns = ["Null Count"])
    if (null_df["Null Count"].sum() == 0):
        return render_template("missvalue.html", dataset = "No Missing Values Found")
    null_df=null_df[null_df["Null Count"] != 0]
    null_df["Null Percentage"] = (null_df["Null Count"] / len(df)) * 100
    null_df["Null Percentage"] = null_df["Null Percentage"].round(2)
    plt.clf()
    null_df["Null Count"].plot(kind="bar", title = "Bar Plot",
                           ylabel="Miss Value Count", color = "g")   
    plt.savefig("static/images/miss/miss_bar.png", bbox_inches ="tight")
    plt.clf() 
    null_df_copy = null_df.copy()
    null_df = null_df.sort_values("Null Count", ascending = False)
    message = "Your dataset has " + str(len(null_df)) + " features with missing values out of " + str(len(df.columns)) + " features."
    null_df.loc["Total"] = null_df.sum()
    
    
    # Imputation Technique through median of feature having no missing values and only few unique values
    feat_list = df.nunique().to_list()
    feat_list_idx = []
    for i in range(len(feat_list)):
        if(feat_list[i] > 1 and feat_list[i] < 10):
            feat_list_idx.append(i)
    feat_list = [df.columns.to_list()[i] for i in feat_list_idx] # Feature list having less unique values    
    return render_template("missvalue.html", dataset = null_df.to_html(), message = message, bar_url = "static/images/miss/miss_bar.png", features = feat_list)


@app.route("/boxplots", methods = ["POST"])
def boxplots():
    global select_list
    select_list = request.form.getlist("columns") # Feature list selected by user
    
    if(len(select_list) != 1):
        return render_template("missvalue2.html", message="Please select exactly one feature")
    x=df.isnull().sum().to_list()
    count=0
    for i in range(len(x)):
        if(x[i]==0):
            count += 1
    x=null_df_copy.index.to_list()
    
    for i in range(len(x)):
        plt.figure(figsize=(15,10))
        sns.boxplot(x=df[select_list[0]], y=df[x[i]], data=df, palette = "winter")
        plt.savefig(f"static/images/miss/boxplot{i}.png", bbox_inches ="tight")
        plt.clf()
    images = [f"static/images/miss/boxplot{i}.png" for i in range(len(x))]
        
    return render_template("missvalue2.html", length = len(x), images=images, message = "BoxPlots to see the outliers!", columns = x)
    
@app.route("/show_miss")
def show_miss():
    return render_template("miss_dataset.html", dataset = df[df.isnull().any(axis=1)].replace(np.nan, '', regex=True).to_html())

@app.route("/fill_misses", methods = ["POST"])
def miss_fill():
    features=request.form.getlist("columns")
    features = [i.replace(","," ") for i in features]

    array=list(np.unique(df[select_list[0]]))

    for i in range(len(features)):
        for j in range(len(array)):
            feature = features[i]
            target=array[j]
            median=df[df[select_list[0]]==target][feature].median()
            df[feature].fillna(median,inplace=True)
    
    return redirect(url_for("phase2"))
            
            
@app.route("/phase3")
def phase3():
    return render_template("Encoding.html")


if __name__=="__main__":
    app.run(host="0.0.0.0")
