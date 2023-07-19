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

# To check accuracy of the model
def check_r2_score(y_test, y_pred):
    
    from sklearn.metrics import r2_score
    score = r2_score(y_test, y_pred)
    
    return score

# Linear Regression
def linear_regression(X_train,y_train):
    
    from sklearn.linear_model import LinearRegression
    linear_regressor=LinearRegression()
    linear_regressor.fit(X_train,y_train)
    
    return linear_regressor

# Ridge Regression
def ridge_regression(X_train,y_train):
    
    from sklearn.linear_model import Ridge
    ridge_regressor=Ridge()
    ridge_regressor.fit(X_train,y_train)
    
    return ridge_regressor

# Lasso Regression
def lasso_regression(X_train,y_train):
    
    from sklearn.linear_model import Lasso
    lasso_regressor=Lasso()
    lasso_regressor.fit(X_train,y_train)
    
    return lasso_regressor


# Elastic NET  Regression
def elastic_net_regression(X_train,y_train):
    
    from sklearn.linear_model import ElasticNet
    elastic_net_regressor=ElasticNet()
    elastic_net_regressor.fit(X_train,y_train)
    
    return elastic_net_regressor
    
    
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
    return render_template("upload copy.html")

#Receiving the dataset here
@app.route("/upload2", methods=['POST'])
def upload_dataset():
    global df
    if request.method == "POST":
        file = request.files["file"]
        if file:
            df = pd.read_csv(file)
        
    return redirect(url_for("main_page"))
#Start
@app.route("/main_page", methods=["GET","POST"])
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

# Histogram Generator
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
        
# Missing Value Analysis
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
        if(feat_list[i] > 1 and feat_list[i] < 15):
            feat_list_idx.append(i)
    feat_list = [df.columns.to_list()[i] for i in feat_list_idx] # Feature list having less unique values    
    return render_template("missvalue.html", dataset = null_df.to_html(), message = message, bar_url = "static/images/miss/miss_bar.png", features = feat_list)

# Detecting Outliers Through Boxplots
@app.route("/boxplots", methods = ["POST"])
def boxplots():
    global select_list
    select_list = request.form.getlist("columns") # Feature list selected by user
    select_list = [i.replace(","," ") for i in select_list]
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
        
    return render_template("missvalue2.html", length = len(x), images=images, message = "BoxPlots to see the outliers!", columns_numerical = x)
    
# Dataset Containing rows with missing values only
@app.route("/show_miss")
def show_miss():
    return render_template("miss_dataset.html", dataset = df[df.isnull().any(axis=1)].replace(np.nan, '', regex=True).to_html())

# Missing Value Imputation
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
            
#Encoding Categorical Features
@app.route("/phase3")
def phase3():
    global send
    x=df.dtypes.astype(str).to_list()
    count = 0
    idx=[] #idx of categorical features
    unique_features=[]
    for i in range(len(x)):
        if x[i]=="object":
            count+=1;
            idx.append(i)
            unique_features.append(list(df[df.columns.to_list()[i]].unique()))
    if count==0:
        return render_template("Encoding1.html", message1="No Categorical Features Found",message2="Your can proceed to next step")
    feature_names=[df.columns.to_list()[i] for i in idx] #categorical feature names
    send={} # dictionary of categorical features and their unique values
    for i in range(len(feature_names)):
        send[feature_names[i]]=unique_features[i]
    return render_template("Encoding.html", message1="Your dataset has "+str(count)+" categorical features",message2="Encoding them to Numeric Values here"
                           ,send=send)

@app.route("/encoding",methods=["GET","POST"])
def encode():
    global encodings,feature
    encoded_values=[] #list of encoded values
    array=[] #list of categorical features
    feature=[] #list of categorical features
    for features,values in send.items():
        array.append(send[features])
        encoded_values.append([request.form.get(f"{value}") for value in values])
        
    encoded_values=[[int(x) if x is not None else None for x in sub_list] for sub_list in encoded_values]
    child_list=[] #list of index and encoded values
    x=[sublist for sublist in encoded_values if None not in sublist]
    child_list.append([encoded_values.index(x[0]),x[0]])
    encodings={} #dictionary of encoded values
    for i in range(len(child_list[0][1])):
        encodings[array[child_list[0][0]][i]]=child_list[0][1][i]
    for features in send.keys():
        feature.append(features)
    feature=feature[child_list[0][0]]
    
    return render_template("Encoding2.html",encodings=encodings)

@app.route("/encode_it",methods=["GET","POST"])
def encode_1():
    global df1
    df[feature]=[encodings[x] for x in df[feature]]
    df1= df.copy()
    return redirect(url_for("phase3"))

@app.route("/phase4")
def phase4():
    return render_template("EDA.html")


@app.route("/phase5")
def phase5():
    return render_template("ML_intro.html")

@app.route("/show_tts")
def tts():
    return render_template("tts.html",columns=df.columns.to_list())
    
@app.route("/start_machine", methods = ["GET","POST"])
def start_machine():
    global X_train,X_test,y_train,y_test
    test=request.form.get("test_size")
    problem=request.form.get("problem")
    
    target = request.form.getlist('columns')
    target = [i.replace(","," ") for i in target]
    target=target[0]
    
    training = request.form.getlist('columns1')
    training = [i.replace(","," ") for i in training]
    
    # Separating Independent and Dependent Features
    X = df[training]
    y=df[target]
    
    
    #splitting
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=float(test),random_state=42)
    
    #scaling the data
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    
    
    if problem=="Regression":
        return render_template("regression.html", test_size=test,
                               training=X_train.shape, testing=X_test.shape)
    else:
        return render_template("classification.html", test_size=test,
                               training=X_train.shape, testing=X_test.shape)
    
    
@app.route("/train_reg_models", methods = ["GET","POST"])
def train_reg_models():
    global regression_models, linear_regressor, ridge_regressor, lasso_regressor, elastic_net_regressor
    regression_models=request.form.getlist("regression_models")
    for i in regression_models:
        
        if i == "linear_reg":
            linear_regressor=linear_regression(X_train,y_train)
            
        if i == "ridge_reg":
            ridge_regressor=ridge_regression(X_train,y_train)
            
        if i == "lasso_reg":
            lasso_regressor=lasso_regression(X_train,y_train)
            
        if i == "elastic_net":
            elastic_net_regressor=elastic_net_regression(X_train,y_train)
            
    return render_template("regression2.html")


@app.route("/train_cls_models", methods = ["GET","POST"])
def train_cls_models():
    global classification_models
    classification_models=request.form.getlist("classification_models")
    return render_template("classification.html")
    
    
@app.route("/test_reg_models", methods = ["GET","POST"])
def test_reg_models():
    for i in regression_models:
        
        if i == "linear_reg":
            linear_reg_pred=linear_regressor.predict(X_test)
            linear_reg_score = check_r2_score(y_test,linear_reg_pred)
            
        if i == "ridge_reg":
            ridge_reg_pred=ridge_regressor.predict(X_test)
            ridge_reg_score = check_r2_score(y_test,ridge_reg_pred)
            
        if i == "lasso_reg":
            lasso_reg_pred=lasso_regressor.predict(X_test)
            lasso_reg_score = check_r2_score(y_test,lasso_reg_pred)
            
        if i == "elastic_net":
            elastic_net_pred=elastic_net_regressor.predict(X_test)
            elastic_net_score = check_r2_score(y_test,elastic_net_pred)
            
        
    return render_template("regression2.html",training=X_train.shape,
                           testing=X_test.shape, accuracy_linear_reg = linear_reg_score,accuracy_ridge_reg = ridge_reg_score,
                           accuracy_lasso_reg = lasso_reg_score, accuray_elastic_net=elastic_net_score)
    
    
@app.route("/test_cls_models", methods = ["GET","POST"])
def test_cls_models():
    return render_template("classification2.html",training=X_train.shape,
                           testing=X_test.shape)  
    
    
if __name__=="__main__":
    app.run(host="0.0.0.0")

