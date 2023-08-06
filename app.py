# Required Dependencies
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import io
import warnings
warnings.filterwarnings("ignore")

# Extreme Gradient booost
import xgboost as xbs

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

matplotlib.use('Agg')
plt=matplotlib.pyplot

# This is our main python file that will run the flask app
app = Flask(__name__)


# To scale down training data
def scale_down(X_train, X_test):
    
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    
    return X_train, X_test

# To check accuracy of the regression models
def check_r2_score(y_test, y_pred):
    
    from sklearn.metrics import r2_score
    score = r2_score(y_test, y_pred)
    
    return score

# To check accuracy of classification models
def check_accuracy(y_test, y_pred):
    
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_test, y_pred)
    
    return score

# Linear Regression
def linear_regression(X_train,y_train):
    
    from sklearn.linear_model import LinearRegression
    linear_regressor=LinearRegression()
    linear_regressor.fit(X_train,y_train)
    
    return linear_regressor

# Ridge Regression (L2 Regularization)
def ridge_regression(X_train,y_train):
    
    from sklearn.linear_model import Ridge
    ridge_regressor=Ridge()
    ridge_regressor.fit(X_train,y_train)
    
    return ridge_regressor

# Lasso Regression (L1 Regularization)
def lasso_regression(X_train,y_train):
    
    from sklearn.linear_model import Lasso
    lasso_regressor=Lasso()
    lasso_regressor.fit(X_train,y_train)
    
    return lasso_regressor


# Elastic NET  Regression (L1 + L2 Regularization)
def elastic_net_regression(X_train,y_train):
    
    from sklearn.linear_model import ElasticNet
    elastic_net_regressor=ElasticNet()
    elastic_net_regressor.fit(X_train,y_train)
    
    return elastic_net_regressor

# Decision Tree Regression
def decision_tree_regression(X_train,y_train):
    
    from sklearn.tree import DecisionTreeRegressor
    tree=DecisionTreeRegressor(random_state=42)
    tree.fit(X_train,y_train)
    
    return tree

# Decision Tree Classification
def decision_tree_classification(X_train,y_train):
    
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train,y_train)
    
    return tree

# Extra Tree Regression
def extra_tree_regression(X_train,y_train):
    
    from sklearn.tree import ExtraTreeRegressor
    trees=ExtraTreeRegressor(random_state=42)
    trees.fit(X_train,y_train)
    
    return trees

# Extra Tree Classification
def extra_tree_classification(X_train,y_train):
    
    from sklearn.tree import ExtraTreeClassifier
    trees=ExtraTreeClassifier(random_state=42)
    trees.fit(X_train,y_train)
    
    return trees
    
# Logistic Regression
def logistic_regression(X_train,y_train,types, random_state, max_iter, multiclass, bias, solver):
    
    if bias == "None":
        bias = None
    
    if types=="Binary":
        from sklearn.linear_model import LogisticRegression
        log_reg=LogisticRegression(random_state=random_state, max_iter=max_iter, penalty=bias, solver=solver)
        log_reg.fit(X_train,y_train)
        return log_reg
   
    if types=="MultiClass":
        from sklearn.linear_model import LogisticRegression
        log_reg=LogisticRegression(random_state=random_state, max_iter=max_iter, multi_class=multiclass, penalty=bias, solver=solver)
        log_reg.fit(X_train,y_train)
        return log_reg
    
#Naive Bias Classifier
def naive_bayes_classifier(X_train,y_train,types):
        
        if types=="Gaussian":
            from sklearn.naive_bayes import GaussianNB
            naive=GaussianNB()
            naive.fit(X_train,y_train)
            return naive
        
        if types=="Multinomial":
            from sklearn.naive_bayes import MultinomialNB
            naive=MultinomialNB()
            naive.fit(X_train,y_train)
            return naive
        
        if types=="Bernoulli":
            from sklearn.naive_bayes import BernoulliNB
            naive=BernoulliNB()
            naive.fit(X_train,y_train)
            return naive
        
        if types=="Complement":
            from sklearn.naive_bayes import ComplementNB
            naive=ComplementNB()
            naive.fit(X_train,y_train)
            return naive
        
        if types=="Categorical":
            from sklearn.naive_bayes import CategoricalNB
            naive=CategoricalNB()
            naive.fit(X_train,y_train)
            return naive
        
#Support Vector Classification
def support_vector_classification(X_train,y_train,random_state,max_iter,kernel,parameter,gamma):
    
    from sklearn.svm import SVC
    svc = SVC(kernel=kernel, C=parameter, gamma=gamma, random_state=random_state, max_iter=max_iter)
    svc.fit(X_train,y_train)
    
    return svc

#Support Vector Regression
def support_vector_regression(X_train,y_train,epsilon,max_iter,kernel,parameter,gamma):
    
    from sklearn.svm import SVR
    svr = SVR(kernel=kernel,epsilon=epsilon, C=parameter, gamma=gamma, max_iter=max_iter)
    svr.fit(X_train,y_train)
    
    return svr

# Random Forest Classification
def random_forest_classification(X_train,y_train,n_estimators,max_depth,max_features,criterion,bootstrap,oob_score):
    
    if max_depth == "None":
        max_depth = None 
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, criterion=criterion, bootstrap=bootstrap, oob_score=oob_score)
    forest.fit(X_train,y_train)
    
    return forest

# Random Forest Regression
def random_forest_regression(X_train,y_train,n_estimators,max_depth,max_features,criterion,bootstrap,oob_score):
    
    if max_depth == "None":
        max_depth = None 
    from sklearn.ensemble import RandomForestRegressor
    forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, criterion=criterion, bootstrap=bootstrap, oob_score=oob_score)
    forest.fit(X_train,y_train)
    
    return forest

# Adaboost Classification
def adaboost_classification(X_train,y_train,n_estimators,learning_rate,algorithm):
    
    from sklearn.ensemble import AdaBoostClassifier
    adaboost = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm)
    adaboost.fit(X_train,y_train)
    
    return adaboost
    
# Gradient Boost Classification
def gradientboost_classification(X_train,y_train,n_estimators,learning_rate,max_depth,criterion):
    
    from sklearn.ensemble import GradientBoostingClassifier
    gradient_boost = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,criterion=criterion)
    gradient_boost.fit(X_train,y_train)
    
    return gradient_boost

# Adaboost Regression
def adaboost_regression(X_train,y_train,n_estimators,learning_rate,loss):
    
    from sklearn.ensemble import AdaBoostRegressor
    adaboost = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, loss=loss)
    adaboost.fit(X_train,y_train)
    
    return adaboost

# Gradient Boost Regressor
def gradient_boost_regression(X_train,y_train,n_estimators, learning_rate, loss, criterion, max_depth, max_features):
    
    from sklearn.ensemble import GradientBoostingRegressor
    gradient_boost = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, criterion=criterion, max_depth=max_depth, max_features=max_features)
    gradient_boost.fit(X_train,y_train)
    
    return gradient_boost

# XGBoost Regressor
def xgboost_regression(X_train,y_train):
    
    xgb_reg=xbs.XGBRegressor()
    xgb_reg.fit(X_train,y_train)
    
    return xgb_reg

# KNN Classifier
def knn_classification(X_train,y_train,n_neighbors,weights,algorithm,leaf_size,p):
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)
    knn.fit(X_train,y_train)
    
    return knn

# KNN Regressor
def knn_regression(X_train,y_train,n_neighbors,weights,algorithm,leaf_size,p):
    
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)
    knn.fit(X_train,y_train)
    
    return knn


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
    plt.figure(figsize=(12,12))
    sns.heatmap(num.corr().round(2), annot=True, cmap="rainbow")
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
    
    
    if(len(arr) > 9):
        return render_template("visualization2.html", message = "Select maximum 9 features")
    if len(arr) != 0:
        sns.set_style("darkgrid")
        
        if(len(arr) == 1):
            plt.figure(figsize=(12,12))
            sns.histplot(x = df[arr[0]], bins = 30, kde = True, color = "r")
        
        elif(len(arr) == 2):
            fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15,10))
            for i in range(len(arr)):
                if (i == 0):
                    sns.histplot(x = df[arr[0]], ax = axes[0], bins = 30, kde = True, color = "r")
                if (i == 1):
                    sns.histplot(x = df[arr[1]], ax = axes[1], bins = 30, kde = True, color = "b")
        
        elif(len(arr) > 2 and len(arr) < 5):
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
        elif(len(arr) == 6):
            fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (15,10))
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
    global null_df_copy,y
    
    null_df = pd.DataFrame(df.isnull().sum().astype(int), columns = ["Null Count"])
    if (null_df["Null Count"].sum() == 0):
        return render_template("missvalue.html", dataset = "No Missing Values Found")
    null_df=null_df[null_df["Null Count"] != 0]
    null_df["Null Percentage"] = (null_df["Null Count"] / len(df)) * 100
    null_df["Null Percentage"] = null_df["Null Percentage"].round(2)
    plt.clf()
    null_df["Null Count"].plot(kind="bar", title = "Bar Plot",
                           ylabel="Miss Value Count", color = "b")   
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
    
    flag = False
    feat = []
    for i in range(len(feat_list)):
        if df[null_df.T.columns.to_list()[i]].dtype == "object":
            flag = True
            feat.append(null_df.T.columns.to_list()[i])
        
        
    
    for i in feat:
        feat_list.remove(i)
        
    temp=null_df_copy.T.columns.to_list()
    y=[]
    
    for i in df.select_dtypes(include=["object"]).columns.to_list():
        if i in temp:
            y.append(i)
            
    return render_template("missvalue.html", dataset = null_df.to_html(), message = message, bar_url = "static/images/miss/miss_bar.png", features = feat_list)

# Detecting Outliers Through Boxplots
@app.route("/boxplots", methods = ["POST"])
def boxplots():
    global select_list,x
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
            
    for i in y:
        if i in x:
            x.remove(i)   
    
        
    return render_template("missvalue2.html", length = len(x), images=images, message = "BoxPlots to see the outliers!", columns_numerical = x)
    
# Dataset Containing rows with missing values only
@app.route("/show_miss")
def show_miss():
    return render_template("miss_dataset.html", dataset = df[df.isnull().any(axis=1)].replace(np.nan, '', regex=True).to_html())

# Missing Value Imputation
@app.route("/fill_misses_numerical", methods = ["POST"])
def miss_fill():
    features=request.form.getlist("columns_num")
    features = [i.replace(","," ") for i in features]

    array=list(np.unique(df[select_list[0]]))

    for i in range(len(features)):
        for j in range(len(array)):
            feature = features[i]
            target=array[j]
            median=df[df[select_list[0]]==target][feature].median()
            df[feature].fillna(median,inplace=True)
    plt.clf()
    return redirect(url_for("phase2"))

@app.route("/fill_misses_categorical", methods = ["GET","POST"])
def fill_misses_categorical():
    features = y
    for i in features:
    
        mode_value = df[i].mode().iloc[0]
        df[i] = df[i].fillna(mode_value)

        
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

@app.route("/download")
def download():
    csv_file=io.StringIO()
    df.to_csv(csv_file,index=False)
    
    return send_file(
        io.BytesIO(csv_file.getvalue().encode()),
        as_attachment=True,
        download_name="Dataset.csv",
        mimetype='text/csv'
    )

@app.route("/phase4")
def phase4():
    return render_template("EDA.html")


@app.route("/phase5")
def phase5():
    
    return render_template("ML_intro.html")

@app.route("/show_tts")
def tts():
    # global df
    # df=pd.read_csv("Real estate.csv")
    return render_template("tts.html",columns=df.columns.to_list())
    
@app.route("/start_machine", methods = ["GET","POST"])
def start_machine():
    global X_train,X_test,y_train,y_test,training,target
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
    
    
    if problem=="Regression":
        return render_template("regression.html", test_size=test,
                               training=X_train.shape, testing=X_test.shape)
    else:
        return render_template("classification.html", test_size=test,
                               training=X_train.shape, testing=X_test.shape)
    
    
@app.route("/train_reg_models", methods = ["GET","POST"])
def train_reg_models():
    global regression_models
    regression_models=request.form.getlist("regression_models")
    
    if len(regression_models) > 1:
        return render_template("regression2.html",training=X_train.shape, testing=X_test.shape)
    
    for i in regression_models:
        
        if i == "linear_reg":
            return render_template("models/LinearRegression/LinearRegression.html",
                                   target=target, trains=training)
        if i == "decision_tree_reg":
            return render_template("models/DecisionTree/DecisionTreeRegressor.html",
                                   target=target, trains=training)
        if i == "svr":
            return render_template("models/SupportVectorMachines/SupportVectorRegressor.html",
                                   target=target, trains=training)
        if i == "random_forest_reg":
            return render_template("models/RandomForest/RandomForestRegressor.html",
                                   target=target, trains=training)
        if i == "adaboost_reg":
            return render_template("models/Boosting/Regressors/AdaboostRegressor.html",
                                   target=target, trains=training)
        if i == "gradientboost_reg":
            return render_template("models/Boosting/Regressors/GradientBoostRegressor.html",
                                   target=target, trains=training)
        if i == "xgboost_reg":
            return render_template("models/Boosting/Regressors/XgboostRegressor.html",
                                   target=target, trains=training)
            
        if i == "knn_reg":
            return render_template("models/KNearestNeighbours/KNNRegressor.html",
                                   target=target, trains=training)
        

@app.route("/train_linear_reg", methods = ["GET","POST"])
def train_linear_reg():
    global linear_regressor
    bias=request.form.get("bias")
    
    # is_scale=request.form.get("scaler")
    # if is_scale=="yes":
    #     X_train,X_test=scale_down(X_train,X_test)
    # Above piece of Code is not Working and i do not know why
    
    if bias == "L1 Regularization":
        linear_regressor=lasso_regression(X_train,y_train)
        return render_template("models/LinearRegression/LinearRegression.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           message="Click Here",columns = training,model = "linear_reg")
        
    elif bias == "L2 Regularization":
        linear_regressor=ridge_regression(X_train,y_train)
        return render_template("models/LinearRegression/LinearRegression.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           message="Click Here",columns = training,model = "linear_reg")
    elif bias == "Both":
        linear_regressor=ridge_regression(X_train,y_train)
        return render_template("models/LinearRegression/LinearRegression.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           message="Click Here",columns = training,model = "linear_reg")
    else:
        linear_regressor=linear_regression(X_train,y_train)
        return render_template("models/LinearRegression/LinearRegression.html",
                            target=target, trains=training,train_status="Model is trained Successfully",
                            columns = training,model = "linear_reg")

@app.route("/test_linear_reg", methods = ["GET","POST"])
def test_linear_reg():
    
    score=check_r2_score(y_test,linear_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})
                   
@app.route("/visualize_linear_reg", methods = ["GET","POST"])
def visualize_linear_reg():
    plt.clf()
    plt.figure(figsize=(15,15))
    plt.scatter(X_train,y_train,color="red",s=2)
    plt.plot(X_train,linear_regressor.predict(X_train),color="blue")
    plt.title("Linear Regression")
    plt.xlabel("Independent Variable")
    plt.ylabel("Dependent Variable")
    plt.savefig("static/images/models/LinearRegression/linear_reg.png", bbox_inches = 'tight')
    
    return render_template("models/LinearRegression/LinearRegression2.html",
                           graph1="static/images/models/LinearRegression/linear_reg.png")


@app.route("/train_decision_tree_reg", methods = ["GET","POST"])
def train_decision_tree_reg():

    global decision_tree_regressor
    tree=request.form.get("tree")
    
    if tree == "ExtraTreeRegressor":
        decision_tree_regressor=extra_tree_regression(X_train,y_train)
        return render_template("models/DecisionTree/DecisionTreeRegressor.html",
                            target=target, trains=training,train_status="ExtraTree is trained Successfully")    
    else:
        decision_tree_regressor=decision_tree_regression(X_train,y_train)
        return render_template("models/DecisionTree/DecisionTreeRegressor.html",
                            target=target, trains=training,train_status="Model is trained Successfully",
                            columns=training,model="decision_tree_reg")
        
@app.route("/test_decision_tree_reg", methods = ["GET","POST"])
def test_decision_tree_reg():
    
    score=check_r2_score(y_test,decision_tree_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_support_vector_regressor", methods = ["GET","POST"])
def train_support_vector_regressor():
    global support_vector_regressor
    
   
    epsilon = request.form.get("epsilon")
    max_iter = request.form.get("max_iter")
    kernel = request.form.get("kernel")
    parameter = request.form.get("parameter")
    gamma = request.form.get("gamma")
    
    if not epsilon:
        epsilon=0.1
    else:
        epsilon = float(epsilon)
        
    if not max_iter:
        max_iter=-1
    else:
        max_iter = int(max_iter)

    if not kernel:
        kernel = "rbf"
        
    if not parameter:
        parameter = 1.0
    else:
        parameter = float(parameter)
        
    if not gamma:
        gamma = "scale"
    elif gamma == "auto":
        gamma = "auto"
    else:
        gamma = float(gamma)
    
    support_vector_regressor = support_vector_regression(X_train,y_train,epsilon=epsilon, max_iter=max_iter, kernel=kernel, parameter=parameter, gamma=gamma)
    return render_template("models/SupportVectorMachines/SupportVectorRegressor.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="support_vector_reg")

@app.route("/test_support_vector_regressor", methods = ["GET","POST"])
def test_support_vector_regressor():
    
    score=check_r2_score(y_test,support_vector_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_random_forest_regressor", methods = ["GET","POST"])
def train_random_forest_regressor():
    global random_forest_regressor
    
    n_estimators = request.form.get("n_estimators")
    max_depth = request.form.get("max_depth")
    max_features = request.form.get("max_features")
    criterion = request.form.get("criterion")
    bootstrap = request.form.get("bootstrap")
    oob_score = request.form.get("oob")
    
    
    if not n_estimators:
        n_estimators=100
    else:
        n_estimators = int(n_estimators)
        
    if not max_depth:
        max_depth=None
    else: 
        max_depth = int(max_depth)
        
    if not max_features:
        max_features=1
    elif max_features == "log2":
        max_features = "log2"
    elif max_features == "None":
        max_features = None
    elif max_features == "sqrt":
        max_features = "sqrt"
    else:
        max_features = float(max_features)
        
    if not criterion:
        criterion="squared_error"
    
    if not bootstrap:
        bootstrap=True
    else:
        bootstrap=False
        
    if not oob_score:
        oob_score=False
    else:
        oob_score=True
    
    random_forest_regressor = random_forest_regression(X_train,y_train,n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, criterion=criterion, bootstrap=bootstrap, oob_score=oob_score)
    return render_template("models/RandomForest/RandomForestRegressor.html",
                           training=X_train.shape, target=X_test.shape,train_status="Model is trained Successfully",
                           columns=training,model="random_forest_reg")

@app.route("/test_random_forest_regressor", methods = ["GET","POST"])
def test_random_forest_regressor():
    
    score=check_r2_score(y_test,random_forest_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_adaboost_regressor", methods = ["GET","POST"])
def train_adaboost_regressor():
    global adaboost_regressor
    
    n_estimators = request.form.get("n_estimators")
    learning_rate = request.form.get("learning_rate")
    loss = request.form.get("loss")
    
    
    if not n_estimators:
        n_estimators=50
    else:
        n_estimators = int(n_estimators)
        
    if not learning_rate:
        learning_rate=1.0
    else:
        learning_rate = float(learning_rate)
        
    if not loss:
        loss="linear"
        
    
    
    adaboost_regressor = adaboost_regression(X_train,y_train,n_estimators=n_estimators, learning_rate=learning_rate, loss=loss)
    return render_template("models/Boosting/Regressors/AdaboostRegressor.html",
                           training=X_train.shape, target=X_test.shape,train_status="Model is trained Successfully",
                           columns=training,model="adaboost_reg")

@app.route("/test_adaboost_regressor", methods = ["GET","POST"])
def test_adaboost_regressor():
    
    score=check_r2_score(y_test,adaboost_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_gradient_boost_regressor", methods = ["GET","POST"])
def train_gradient_boost_regressor():
    global gradient_boost_regressor
    
    n_estimators = request.form.get("n_estimators")
    learning_rate = request.form.get("learning_rate")
    loss = request.form.get("loss")
    criterion = request.form.get("criterion")
    max_depth = request.form.get("max_depth")
    max_features = request.form.get("max_features")
    
    
    if not n_estimators:
        n_estimators=100
    else:
        n_estimators = int(n_estimators)
        
    if not learning_rate:
        learning_rate=0.1
    else:
        learning_rate = float(learning_rate)
        
    if not loss:
        loss="squared_error"
        
    if not criterion:
        criterion="friedman_mse"
        
    if not max_depth:
        max_depth=3
    else:
        max_depth = int(max_depth)
        
    if not max_features:
        max_features=None
    elif max_features == "log2":
        max_features = "log2"
    elif max_features == "None":
        max_features = None
    elif max_features == "sqrt":
        max_features = "sqrt"
    else:
        max_features = float(max_features)
        
    gradient_boost_regressor = gradient_boost_regression(X_train,y_train,n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, criterion=criterion, max_depth=max_depth, max_features=max_features)
    return render_template("models/Boosting/Regressors/GradientBoostRegressor.html",
                           training=X_train.shape, target=X_test.shape,train_status="Model is trained Successfully",
                           columns=training,model="gradient_boost_reg")

@app.route("/test_gradient_boost_regressor", methods = ["GET","POST"])
def test_gradient_boost_regressor():
    
    score=check_r2_score(y_test,gradient_boost_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_xgboost_regressor", methods = ["GET","POST"])
def train_xgboost_regressor():
    global xgboost_regressor
    
    xgboost_regressor=xgboost_regression(X_train,y_train)
    return render_template("models/Boosting/Regressors/XgboostRegressor.html",
                           training=X_train.shape, target=X_test.shape,train_status="Model is trained Successfully",
                           columns=training,model="xgboost_reg")
    
@app.route("/test_xgboost_regressor", methods = ["GET","POST"])
def test_xgboost_regressor():
        
        score=check_r2_score(y_test,xgboost_regressor.predict(X_test))
        score=score*100
        return jsonify({"score":score})
    
@app.route("/train_knn_regressor", methods = ["GET","POST"])
def train_knn_regressor():
    global knn_regressor
    
    n_neighbors = request.form.get("n_neighbors")
    weights = request.form.get("weights")
    algorithm = request.form.get("algorithm")
    leaf_size = request.form.get("leaf_size")
    p = request.form.get("p")
    
    if not n_neighbors:
        n_neighbors=5
    else:
        n_neighbors = int(n_neighbors)
        
    if not weights:
        weights="uniform"
    
        
    if not algorithm:
        algorithm="auto"
        
    if not leaf_size:
        leaf_size=30
    else:
        leaf_size = int(leaf_size)
        
    if not p:
        p=2
    else:
        p = int(p)
        
    knn_regressor=knn_regression(X_train,y_train,n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)
    return render_template("models/KNearestNeighbours/KNNRegressor.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model = "knn_reg")
    
@app.route("/test_knn_regressor", methods = ["GET","POST"])
def test_knn_regressor():
        
        score=check_r2_score(y_test,knn_regressor.predict(X_test))
        score=score*100
        return jsonify({"score":score})

    
@app.route("/test_reg_models", methods = ["GET","POST"])
def test_reg_models():
    
    for i in regression_models:
        
        if i == "linear_reg":
            lin_reg = LinearRegression()
            lin_reg.fit(X_train,y_train)
            accuracy_linear_reg = check_r2_score(y_test,lin_reg.predict(X_test))
            accuracy_linear_reg = accuracy_linear_reg*100
            
        elif i == "decision_tree_reg":
            dt_reg = DecisionTreeRegressor()
            dt_reg.fit(X_train,y_train)
            accuracy_decision_tree_reg=check_r2_score(y_test,dt_reg.predict(X_test))
            accuracy_decision_tree_reg=accuracy_decision_tree_reg*100
            
        elif i == "svr":
            svr = SVR()
            svr.fit(X_train,y_train)
            accuracy_svr=check_r2_score(y_test,svr.predict(X_test))
            accuracy_svr=accuracy_svr*100
            
        elif i == "random_forest_reg":
            rf_reg = RandomForestRegressor()
            rf_reg.fit(X_train,y_train)
            accuracy_random_forest_reg=check_r2_score(y_test,rf_reg.predict(X_test))
            accuracy_random_forest_reg=accuracy_random_forest_reg*100
            
        elif i == "adaboost_reg":
            ada_reg = AdaBoostRegressor()
            ada_reg.fit(X_train,y_train)
            accuracy_adaboost_reg=check_r2_score(y_test,ada_reg.predict(X_test))
            accuracy_adaboost_reg=accuracy_adaboost_reg*100
            
        elif i == "gradientboost_reg":
            gb_reg = GradientBoostingRegressor()
            gb_reg.fit(X_train,y_train)
            accuracy_gradient_boost_reg=check_r2_score(y_test,gb_reg.predict(X_test))
            accuracy_gradient_boost_reg=accuracy_gradient_boost_reg*100
            
        elif i == "xgboost_reg":
            xgb_reg = xbs.XGBRegressor()
            xgb_reg.fit(X_train,y_train)
            accuracy_xgboost_reg=check_r2_score(y_test,xgb_reg.predict(X_test))
            accuracy_xgboost_reg=accuracy_xgboost_reg*100
            
        elif i == "knn_reg":
            knn_reg = KNeighborsRegressor()
            knn_reg.fit(X_train,y_train)
            accuracy_knn_reg=check_r2_score(y_test,knn_reg.predict(X_test))
            accuracy_knn_reg=accuracy_knn_reg*100
    
    return render_template("regression2.html",training=X_train.shape, testing=X_test.shape,
                           accuracy_linear_reg=accuracy_linear_reg,
                           accuracy_decision_tree_reg=accuracy_decision_tree_reg,
                           accuracy_svr=accuracy_svr,
                           accuracy_random_forest_reg=accuracy_random_forest_reg,
                           accuracy_adaboost_reg=accuracy_adaboost_reg,
                           accuracy_gradient_boost_reg=accuracy_gradient_boost_reg,
                           accuracy_xgboost_reg=accuracy_xgboost_reg,
                           accuracy_knn_reg=accuracy_knn_reg)
            
@app.route("/train_cls_models", methods = ["GET","POST"])
def train_cls_models():
    global classification_models
    classification_models=request.form.getlist("classification_models")
    
    if len(classification_models) > 1:
        return render_template("classification2.html",training=X_train.shape, testing=X_test.shape)
    
    for i in classification_models:
        
        if i == "decision_tree_cls":
            return render_template("models/DecisionTree/DecisionTreeClassifier.html",
                                   target=target, trains=training)
        if i== "logistic":
            return render_template("models/LogisticalRegression/Logistic.html",
                                      target=target, trains=training)
        if i== "naive_bayes":
            return render_template("models/NaiveBayes/NaiveBayes.html",
                                      target=target, trains=training)
        if i == "svc":
            return render_template("models/SupportVectorMachines/SupportVectorClassifier.html",
                                      target=target, trains=training)
        if i == "random_forest_cls":
            return render_template("models/RandomForest/RandomForestClassifier.html",
                                      target=target, trains=training)
        if i == "adaboost":
            return render_template("models/Boosting/Classifiers/AdaboostClassifier.html",
                                      target=target, trains=training)
        if i == "gradientboost":
            return render_template("models/Boosting/Classifiers/GradientBoostClassifier.html",
                                      target=target, trains=training)
        if i == "knn_cls":
            return render_template("models/KNearestNeighbours/KNNClassifier.html",
                                      target=target, trains=training)
            
@app.route("/train_logistic_regression_classifier", methods = ["GET","POST"])
def train_logistic_regression_classifier():
    global logistic_regression_classifier
    
    classify = request.form.get("logistic")
    random_state = request.form.get("random_state")
    max_iter = request.form.get("max_iter")
    multiclass = request.form.get("multiclass")
    bias = request.form.get("bias")
    solver = request.form.get("solver")
    
    
    if not random_state:
        random_state=None
    else:
        random_state = int(random_state)
        
        
    if not max_iter:
        max_iter=100
    else:
        max_iter = int(max_iter)

    if not multiclass:
        multiclass="auto"
    if not bias:
        bias = "l2"
    if not solver:
        solver="lbfgs"
        
    
    
    logistic_regression_classifier=logistic_regression(X_train,y_train,types=classify, random_state=random_state, max_iter=max_iter, multiclass=multiclass, bias=bias, solver=solver)
    return render_template("models/LogisticalRegression/Logistic.html",
                            target=target, trains=training,train_status=f"{classify} Logistic Model is trained Successfully",
                            columns=training,model="logistic_cls")
    

@app.route("/test_logistical_regression_classifier", methods = ["GET","POST"])
def test_logistical_regression_classifier():
    
    score=check_accuracy(y_test,logistic_regression_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_naive_bayes_classifier", methods = ["GET","POST"])
def train_native_bayes_classifier():
    global native_bayes_classifier
    classify = request.form.get("algos")
    native_bayes_classifier=naive_bayes_classifier(X_train,y_train,types=classify)
    return render_template("models/NaiveBayes/NaiveBayes.html",
                            target=target, trains=training,train_status=f"{classify} Naive Bayes Model is trained Successfully",
                            columns=training,model="naive_bayes")    

@app.route("/test_naive_bayes_classifier", methods = ["GET","POST"])
def test_native_bayes_classifier():
        
        score=check_accuracy(y_test,native_bayes_classifier.predict(X_test))
        score=score*100
        return jsonify({"score":score})

@app.route("/train_decision_tree_classifier", methods = ["GET","POST"])
def train_decision_tree_classifier():
    global decision_tree_classifier
    tree = request.form.get("tree")
    
    if tree == "ExtraTreeClassifier":
        decision_tree_classifier=extra_tree_classification(X_train,y_train)
        return render_template("models/DecisionTree/DecisionTreeClassifier.html",
                                target=target, trains=training,train_status="Extra Tree Model is trained Successfully",
                                columns=training,model="decision_tree_cls")
    else:
        decision_tree_classifier=decision_tree_classification(X_train,y_train)
        return render_template("models/DecisionTree/DecisionTreeClassifier.html",
                                target=target, trains=training,train_status="Model is trained Successfully",
                                columns=training,model="decision_tree_cls")

@app.route("/test_decision_tree_classifier", methods = ["GET","POST"])
def test_decision_tree_classifier():
    
    score=check_accuracy(y_test,decision_tree_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_support_vector_classifier", methods = ["GET","POST"])
def train_support_vector_classifier():
    global support_vector_classifier
    
   
    random_state = request.form.get("random_state")
    max_iter = request.form.get("max_iter")
    kernel = request.form.get("kernel")
    parameter = request.form.get("parameter")
    gamma = request.form.get("gamma")
    
    if not random_state:
        random_state=None
    else:
        random_state = int(random_state)
        
    if not max_iter:
        max_iter=-1
    else:
        max_iter = int(max_iter)

    if not kernel:
        kernel = "rbf"
        
    if not parameter:
        parameter = 1.0
    else:
        parameter = float(parameter)
        
    if not gamma:
        gamma = "scale"
    elif gamma == "auto":
        gamma = "auto"
    else:
        gamma = float(gamma)
    
    support_vector_classifier = support_vector_classification(X_train,y_train,random_state=random_state, max_iter=max_iter, kernel=kernel, parameter=parameter, gamma=gamma)
    return render_template("models/SupportVectorMachines/SupportVectorClassifier.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="svc")

@app.route("/test_support_vector_classifier", methods = ["GET","POST"])
def test_support_vector_classifier():
    
    score=check_accuracy(y_test,support_vector_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_random_forest_classifier", methods = ["GET","POST"])
def train_random_forest_classifier():
    global random_forest_classifier
    
    n_estimators = request.form.get("n_estimators")
    max_depth = request.form.get("max_depth")
    max_features = request.form.get("max_features")
    criterion = request.form.get("criterion")
    bootstrap = request.form.get("bootstrap")
    oob_score = request.form.get("oob")
    
    
    if not n_estimators:
        n_estimators=100
    else:
        n_estimators = int(n_estimators)
        
    if not max_depth:
        max_depth=None
    else: 
        max_depth = int(max_depth)
        
    if not max_features:
        max_features="sqrt"
    elif max_features == "log2":
        max_features = "log2"
    elif max_features == "None":
        max_features = None
    else:
        max_features = float(max_features)
        
    if not criterion:
        criterion="gini"
    
    if not bootstrap:
        bootstrap=True
    else:
        bootstrap=False
        
    if not oob_score:
        oob_score=False
    else:
        oob_score=True
        
    random_forest_classifier=random_forest_classification(X_train,y_train,n_estimators=n_estimators, max_depth=max_depth,max_features=max_features, criterion=criterion, bootstrap=bootstrap, oob_score=oob_score)
    return render_template("models/RandomForest/RandomForestClassifier.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="random_forest_cls")

@app.route("/test_random_forest_classifier", methods = ["GET","POST"])
def test_random_forest_classifier():
    
    score=check_accuracy(y_test,random_forest_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_adaboost_classifier", methods = ["GET","POST"])
def train_adaboost_classifier():
    global adaboost_classifier
    
    n_estimators = request.form.get("n_estimators")
    learning_rate = request.form.get("learning_rate")
    algorithm = request.form.get("algorithm")
    
    if not n_estimators:
        n_estimators=50
    else:
        n_estimators = int(n_estimators)
        
    if not learning_rate:
        learning_rate=1.0
    else:
        learning_rate = float(learning_rate)
        
    if not algorithm:
        algorithm="SAMME.R"
    
    adaboost_classifier=adaboost_classification(X_train,y_train,n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm)
    return render_template("models/Boosting/Classifiers/AdaboostClassifier.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="adaboost_cls")

@app.route("/test_adaboost_classifier", methods = ["GET","POST"])
def test_adaboost_classifier():
    
    score=check_accuracy(y_test,adaboost_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_gradientboost_classifier", methods = ["GET","POST"])
def train_gradientboost_classifier():
    global gradientboost_classifier
    
    n_estimators = request.form.get("n_estimators")
    learning_rate = request.form.get("learning_rate")
    max_depth = request.form.get("max_depth")
    criterion = request.form.get("criterion")
    
    
    if not n_estimators:
        n_estimators=100
    else:
        n_estimators = int(n_estimators)
        
    if not learning_rate:
        learning_rate=0.1
    else:
        learning_rate = float(learning_rate)
        
    if not max_depth:
        max_depth=3
    else:
        max_depth = int(max_depth)
        
    if not criterion:
        criterion="friedman_mse"
    
    
    gradientboost_classifier=gradientboost_classification(X_train,y_train,n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, criterion=criterion)
    return render_template("models/Boosting/Classifiers/GradientBoostClassifier.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="gradient_boosting_cls")

@app.route("/test_gradientboost_classifier", methods = ["GET","POST"])
def test_gradientboost_classifier():
    
    score=check_accuracy(y_test,gradientboost_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_knn_classifier", methods = ["GET","POST"])
def train_knn_classifier():
    global knn_classifier
    
    n_neighbors = request.form.get("n_neighbours")
    weights = request.form.get("weights")
    algorithm = request.form.get("algorithm")
    leaf_size = request.form.get("leaf_size")
    p = request.form.get("p")
    
    if not n_neighbors:
        n_neighbors=5
    else:
        n_neighbors = int(n_neighbors)
        
    if not weights:
        weights="uniform"
    
        
    if not algorithm:
        algorithm="auto"
        
    if not leaf_size:
        leaf_size=30
    else:
        leaf_size = int(leaf_size)
        
    if not p:
        p=2
    else:
        p = int(p)
        
    knn_classifier=knn_classification(X_train,y_train,n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)
    return render_template("models/KNearestNeighbours/KNNClassifier.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="knn_cls")

@app.route("/test_knn_classifier", methods = ["GET","POST"])
def test_knn_classifier():
    
    score=check_accuracy(y_test,knn_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})
    

@app.route("/test_cls_models", methods = ["GET","POST"])
def test_cls_models():
    
    for i in classification_models:
        
        if i == "logistic":
            
            if len(y_train.unique()) > 2:
                log_cls = LogisticRegression(multi_class="ovr")
                log_cls.fit(X_train,y_train)
                accuracy_logistic = check_accuracy(y_test,log_cls.predict(X_test))
                accuracy_logistic=accuracy_logistic*100
            else:
                log_cls = LogisticRegression()
                log_cls.fit(X_train,y_train)
                accuracy_logistic = check_accuracy(y_test,log_cls.predict(X_test))
                accuracy_logistic=accuracy_logistic*100
            
        elif i == "decision_tree_cls":
            dt_cls = DecisionTreeClassifier()
            dt_cls.fit(X_train,y_train)
            accuracy_decision_tree_cls=check_accuracy(y_test,dt_cls.predict(X_test))
            accuracy_decision_tree_cls=accuracy_decision_tree_cls*100
            
        elif i == "naive_bayes":
            
            if len(y_train.unique()) > 2:
                nb_cls = MultinomialNB()
                nb_cls.fit(X_train,y_train)
                accuracy_naive_bayes=check_accuracy(y_test,nb_cls.predict(X_test))
                accuracy_naive_bayes=accuracy_naive_bayes*100
            else:
                nb_cls = GaussianNB()
                nb_cls.fit(X_train,y_train)
                accuracy_naive_bayes=check_accuracy(y_test,nb_cls.predict(X_test))
                accuracy_naive_bayes=accuracy_naive_bayes*100
        
        elif i == "svc":
            svc_cls = SVC()
            svc_cls.fit(X_train,y_train)
            accuracy_svc=check_accuracy(y_test,svc_cls.predict(X_test))
            accuracy_svc=accuracy_svc*100
        
        elif i == "random_forest_cls":
            rf_cls = RandomForestClassifier()
            rf_cls.fit(X_train,y_train)
            accuracy_random_forest_cls=check_accuracy(y_test,rf_cls.predict(X_test))
            accuracy_random_forest_cls=accuracy_random_forest_cls*100
            
        elif i == "adaboost":
            adaboost_cls = AdaBoostClassifier()
            adaboost_cls.fit(X_train,y_train)
            accuracy_adaboost=check_accuracy(y_test,adaboost_cls.predict(X_test))
            accuracy_adaboost=accuracy_adaboost*100
            
        elif i == "gradientboost":
            gradientboost_cls = GradientBoostingClassifier()
            gradientboost_cls.fit(X_train,y_train)
            accuracy_gradientboost=check_accuracy(y_test,gradientboost_cls.predict(X_test))
            accuracy_gradientboost=accuracy_gradientboost*100
            
        elif i == "knn_cls":
            knn_cls = KNeighborsClassifier()
            knn_cls.fit(X_train,y_train)
            accuracy_knn_cls=check_accuracy(y_test,knn_cls.predict(X_test))
            accuracy_knn_cls=accuracy_knn_cls*100
            
    return render_template("classification2.html",training=X_train.shape, testing=X_test.shape,
                               accuracy_logistic=accuracy_logistic,
                               accuracy_decision_tree_cls=accuracy_decision_tree_cls,
                               accuracy_naive_bayes=accuracy_naive_bayes,
                               accuracy_svc=accuracy_svc,
                               accuracy_random_forest_cls=accuracy_random_forest_cls,
                               accuracy_adaboost=accuracy_adaboost,
                               accuracy_gradientboost=accuracy_gradientboost,
                               accuracy_knn_cls=accuracy_knn_cls)

@app.route("/graph", methods = ["GET","POST"])
def grapher():
    columns = df.columns.to_list()
    return render_template("/Graph/main.html",columns=columns)

@app.route("/plot_graph", methods = ["GET","POST"])
def show_graph():
    
    feature11 = request.form.get("feature11")
    feature12 = request.form.get("feature12")
    feature21 = request.form.get("feature21")
    feature22 = request.form.get("feature22")
    
    if feature11:
        feature11 = feature11.replace(","," ")
    
    if feature12:
        feature12 = feature12.replace(","," ")
        
    if feature21:
        feature21 = feature21.replace(","," ")
        
    if feature22:
        feature22 = feature22.replace(","," ")
        

    plot1 = request.form.get("columns1")
    plot2 = request.form.get("columns2")
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    
    if plot1 == "scatter plot":
        axes[0].scatter(df[feature11],df[feature12])
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)
        
    elif plot1 == "line plot":
        axes[0].plot(df[feature11],df[feature12])
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)
        
    elif plot1 == "bar plot":
        axes[0].bar(df[feature11],df[feature12])
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)
        
    elif plot1 == "box plot":
        axes[0].boxplot(df[feature11])
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)
    
    elif plot1 == "violin plot":
        axes[0].violinplot(df[feature11])
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)   
        
    elif plot1 == "heatmap":
        df1 = df[[feature11,feature12]]
        sns.heatmap(df1.corr(),annot=True, ax = axes[0])
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)
    
    elif plot1 == "hexbin":
        left_hex = axes[0].hexbin(df[feature11],df[feature12],gridsize=20, cmap = "Blues")
        axes[0].set_title(f"Hexbin plot of {feature11} and {feature12}")
        plt.colorbar(left_hex, ax=axes[0], label = "Density")
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)
        
    
        
        
    # Same for plot 2
    if plot2 == "scatter plot":
        axes[1].scatter(df[feature21],df[feature22])
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
        
    elif plot2 == "line plot":
        axes[1].plot(df[feature21],df[feature22])
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
        
    elif plot2 == "bar plot":
        axes[1].bar(df[feature21],df[feature22])
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
        
    elif plot2 == "box plot":
        axes[1].boxplot(df[feature21])
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
    
    elif plot2 == "violin plot":
        axes[1].violinplot(df[feature21])
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
        
    elif plot2 == "heatmap":
        df2 = df[[feature21,feature22]]
        sns.heatmap(df2.corr(),annot=True, ax = axes[1])
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
        
    elif plot2 == "hexbin":
        right_hex = axes[1].hexbin(df[feature21],df[feature22],gridsize=20, cmap = "Reds")
        axes[1].set_title(f"Hexbin plot of {feature21} and {feature22}")
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
        plt.colorbar(right_hex, ax=axes[1], label = "Density")
        
        
    
    
    plt.savefig("static/images/GraphTool/plotter.png", bbox_inches='tight')
    
    return render_template("Graph/graph1.html", graph = "static/images/GraphTool/plotter.png", message = "Graph plotted successfully")


@app.route("/plot_piechart", methods = ["GET","POST"])
def plot_pie():
    
    feature31 = request.form.get("feature31")
    if feature31:
        feature31 = feature31.replace(","," ")
        
    count = df[feature31].value_counts()
    plt.figure(figsize=(10,10))
    plt.pie(count, labels = count.index, autopct='%1.1f%%', shadow=True, startangle=90)
    
    plt.title(f"Pie chart for {feature31}")
    
    plt.savefig("static/images/GraphTool/pie.png", bbox_inches='tight')
    
    return render_template("Graph/graph2.html", graph = "static/images/GraphTool/pie.png", message = "Pie chart plotted successfully")

@app.route("/plot_gbarplot", methods = ["GET","POST"])
def gbarplot():
    
    target_feature = request.form.get("feature41")
    
    if target_feature:
        target_feature = target_feature.replace(","," ")
    
    features = request.form.getlist("feature42")
    features = [feature.replace(","," ") for feature in features]
    
    feature41 = features[0];
    feature42 = features[1];
    
    group_data = df.groupby(target_feature)[[feature41,feature42]].mean()
    
    plt.figure(figsize=(15,15))
    width = 0.4 # width of bar
    
    bar_positions1 = np.arange(len(group_data))
    bar_position2 = bar_positions1 + width
    
    plt.bar(bar_positions1, group_data[feature41], width=width, label=feature41)
    plt.bar(bar_position2, group_data[feature42], width=width, label=feature42)
    
    plt.xticks((bar_positions1+bar_position2)/2, group_data.index)
    plt.xlabel(target_feature)
    plt.ylabel("Average Value")
    plt.title(f"Grouped bar plot for {feature41} and {feature42}")
    plt.legend()
    
    plt.savefig("static/images/GraphTool/gbarplot.png", bbox_inches='tight')
    
    return render_template("Graph/graph3.html", graph = "static/images/GraphTool/gbarplot.png", message = "Grouped bar plot plotted successfully")

# Predictions Regressions
@app.route("/predict_linear_reg", methods = ["GET","POST"])
def predict_linear_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = linear_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Linear Regression", prediction=score[0])    

@app.route("/predict_decision_tree_reg", methods = ["GET","POST"])
def predict_decision_tree_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = decision_tree_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Decision Tree Regression", prediction=score[0])

@app.route("/predict_support_vector_reg", methods = ["GET","POST"])
def predict_svr():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = support_vector_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Support Vector Regression", prediction=score[0])

@app.route("/predict_random_forest_reg", methods = ["GET","POST"])
def predict_random_forest_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = random_forest_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Random Forest Regression", prediction=score[0])

@app.route("/predict_knn_reg", methods = ["GET","POST"])
def predict_knn_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = knn_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "K Nearest Neighbors Regression", prediction=score[0])

@app.route("/predict_adaboost_reg", methods = ["GET","POST"])
def predict_adaboost_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = adaboost_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "AdaBoost Regression", prediction=score[0])

@app.route("/predict_gradient_boost_reg", methods = ["GET","POST"])
def predict_gradient_boosting_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = gradient_boost_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Gradient Boosting Regression", prediction=score[0])

@app.route("/predict_xgboost_reg", methods = ["GET","POST"])
def predict_xgboost_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = xgboost_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "XGBoost Regression", prediction=score[0])

# Prediction classification all models
@app.route("/predict_logistic_cls", methods = ["GET","POST"])
def predict_logistic_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = logistic_regression_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Logistic Regression", prediction=score[0])

@app.route("/predict_knn_cls", methods = ["GET","POST"])
def predict_knn_cls():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = knn_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "K Nearest Neighbors Classifier", prediction=score[0])

@app.route("/predict_svc", methods = ["GET","POST"])
def predict_svc():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = support_vector_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Support Vector Classifier", prediction=score[0])

@app.route("/predict_naive_bayes", methods = ["GET","POST"])
def predict_naive_bayes():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = native_bayes_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Naive Bayes Classifier", prediction=score[0])

@app.route("/predict_decision_tree_cls", methods = ["GET","POST"])
def predict_decision_tree_cls():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = decision_tree_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Decision Tree Classifier", prediction=score[0])

@app.route("/predict_random_forest_cls", methods = ["GET","POST"])
def predict_random_forest_cls():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = random_forest_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Random Forest Classifier", prediction=score[0])

@app.route("/predict_adaboost_cls", methods = ["GET","POST"])
def predict_adaboost_cls():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = adaboost_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "AdaBoost Classifier", prediction=score[0])

@app.route("/predict_gradient_boosting_cls", methods = ["GET","POST"])
def predict_gradient_boosting_cls():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = gradientboost_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Gradient Boosting Classifier", prediction=score[0])

if __name__=="__main__":
    app.run(host="0.0.0.0")

