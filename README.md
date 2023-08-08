# Data Analytical and Machine Learning Tool 
You can see the deployed thing [HERE](https://automaticmltool.onrender.com/) but it may run very slow

## Our Team: 
1. Ayush Raina (Project Lead): handled most of the Backend part of this Project (Indian Institute of Science, Bengaluru, Maths and Computing)
2. Aastha Verma (Design) : handled all the designing(Frontend) of the Project (Indian Institute of Technology, Delhi, Computer Science and Engineering)
3. Srijan : helped in managing tasks and finding errors (Indian Institute of Science, Bengaluru, Maths and Computing)

## Aim
With this tool anyone who does not have any knowledge of Machine Learning will be able to train Machine Learning models and can make predictions on his/her new data. We have tried our best to make the layout of this tool very simple and attractive. You will just require a dataset to get started with. This tool is packed with numerous other features for Data Analysis. You can even plot a number of graphs to visualize the data, its trends and patterns.

## Setup and Installation
1. First of all you need to bring the project to your system, for that you can type this into your terminal,  "git clone https://github.com/ayushraina2028/Data-Analytics-and-Machine-Learning-Tool.git".  After hitting ENTER, whole thing will be cloned into your system.
2. Now its time to install the dependencies that are listed in requirements, you just need to type this in your terminal, "pip install -r requirements.txt". Now all the dependencies will be automatically installed in your system
3. To start with, we have already provided 2 datasets in the files, one is for Classification Problem statement(winequalityN.csv) and other is for Regression Problem statement(real Estate.csv)
4. Now we are ready to get started just run the app.py file by run button in vs code directly or type this into your terminal, "python app.py" and it will run your file, it will take few seconds to run after that it will give 3 links, click on 3rd one in your Browser.


## How to use
In Home Page, our profile links are attached where you can contact us and there is a START button, click that to proceed to next step
In Next Introductory Page, things are mentioned what we will be performing in this tool, click on Lets Start button at end of this page for next.
Now here we need to upload our dataset, we have already provided 2 datasets, you can upload anyone or upload you own dataset also, but make your you upload dataset from trusted sources only, otherwise corrupted dataset will give you many errors
Suppose you started with winequalityN dataset, hit upload and you will reach,

## Phase 1: Data Processing
4 options are given here and 1 option to check dataset that you have uploaded
1. Data Insights I: This will show some very basic metrics regarding your dataset
2. Data Insights II: This will show some statistical calculations regarding the dataset
3. Visualization I: This will first show a co-relation heatmap of your dataset that will show how a feature depends on other feature very beautifully + There is a histogram generator in this part where you can just select you features, and plot upto 9 histograms at a single time that will help you to understand the distribution of your dataset
4. Categorical Analysis: This will first show some metrics related to categorical features in the dataset, then there is also a histogram generator that can be used to understand distribution
5. Now after playing with this, we can move to next step which is Missing Value Analysis

## Phase 2: Missing Value Analysis
1. First of all this will show a table containing Features that have missing values, no of missng values, and percentage also.
2. It will show show a bar plot of features and the no. of missing values in them
3. As missing values as very important to handle, It is very easy to do this from this page, Just click Fill Here option in front of Categorical Features, this will fill all the categorical features at once by the mode value(most occuring value)
4. Next Comes to filling of Numerical Features, you will see some features that needs to be selected for proceeding further, suppose you selected "type" here, it will show you boxplots, in which we can clearly see where median is and all the points outside the box are outliers. after that there are option to fill those. Mechanism of filling is a little bit complicated, what it does is it selects that row in which missing value is present, type value of that row is x suppose so it will select median of all rows have type value x.
5. 

## Encoding Categorical Features
After filling missing values we proceed towards next step which is Encoding Categorical Features, In wine dataset one column in type which has features like red, white, these are called categorical features means which are neither int or float. We need to encode them into numerical ones. In this page you will get input field where you can enter your numeric values for corresponding categorical values, after putting on confirm pages you can confirm changes. To see the changes you can click on Check Dataset Option.

## Visualization Pro Max
Next is Visualization: Here you a wide variety of Graph Plotting things, First things you can 2 plot graphs at a time of over seven kinds as of till now , which contain scatterplot, lineplot, barlot, boxplot, violinplot, heatmap, hexbin. There is more you can even plot a Pie Chart for any feature in your dataset.Wait there is another thing called grouped bar plot, which can be used to visualize target variable with any 2 features. It produces very good visualizations if Data is Good

# Introduction to Machine Learning

## Train Test Split
Now we seen some good insights from our dataset, Lets proceed towards ML
1. First of all we need to Enter Test Size, Please enter a decimal between 0 and 1.
2. After that we need to select Problem Statement Type, Regression or Classification.
3. Now select the Target Feature, which you want to predict
4. Finally Select the Training Features, you can select all at once, or play with it however you want and click on Proceed
5. Depending on what you select, means if you select Classification, you will be shown classification Algorithms, and similar for Regression

## Algorithms
Supposed you selected Regression thing, you will see a list of algorithms.
1. Either Select 1 algorithm or Select all at once for Comparison
2. If you select all algorithms, then start training, you will be shown another page where you can click and see accuracy of all the algorithms, how they performed on test data.
3. If you select one algorithm, it will open a separate page for every algorithm listed there. In all pages you will get a option to train, test, predict. You can click on Train, once model is trained you will see success message, after that you can click of test to check the accuracy of test data.
4. You can even change some parameters that are given on particular pages of algorithms and play with it, and can find if you get increased accuracy by changing something.

## Predictions
1. For prediction you need to select any one algorithm and open its particular page
2. When you open page of algorithm, you will hit start training, When training is completed, you will see input fields after scrolling a little bit.
3. In these input fields you can put your new data and click on Predict. This will show you Prediction based on new data.


That's all from our side for now, Do suggest changes 
Contact me Here
[LinkedIN](https://www.linkedin.com/in/ayushrainaiisc/)
[Email](ayushraina@iisc.ac.in)

 


