import scipy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request, session
import pandas as pd
import os
from werkzeug.utils import secure_filename


# Define folder to save uploaded files to process further
# UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads') #For windows
UPLOAD_FOLDER = os.path.join(
    '/Users/borhan/Desktop/projectMadam/Deployment-flask-master')

# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__, template_folder='templates', static_folder='static/css')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'


@app.route('/')
def index():
    return render_template('InterfaceDesign.html')


@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']

        # Extracting uploaded data file name
        data_filename = secure_filename(uploaded_df.filename)

        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(
            app.config['UPLOAD_FOLDER'], data_filename))

        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(
            app.config['UPLOAD_FOLDER'], data_filename)

        return render_template('InterfaceDesign2.html')
# (1)----------------------------------------- Classification Model ----------------------------------------------#


@app.route('/show_result_classification', methods=["POST"])
def showResult():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)

    # Importing the libraries
    # import pandas as pd

    # Import Decision Tree Classifier

    # Import train_test_split function

    # split dataset in features and target variable
    # feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'Email', 'DiabetesPedigreeFunction']

    # X = uploaded_df[feature_cols]  # Features
    # y = uploaded_df.Outcome  # Target variable

    X = uploaded_df.drop('Outcome', axis=1)
    y = uploaded_df[['Outcome']]

    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    encoder = OrdinalEncoder()
    encoder.fit(X)
    X_encoded = encoder.transform(X)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=1)  # 70% training and 30% test

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # print(y_pred)

    # print(accuracy)
    global accuracyCLS
    accuracyCLS = round(accuracy_score(y_pred, y_test), 4)
    # print(accuracyCLS)

    # print('Recall: %.3f' % recall_score(y_test, y_pred))
    global recallCLS
    recallCLS = round(recall_score(y_test, y_pred), 4)
    # print(('precision_score(y_test, y_pred)))
    global precisionCLS
    precisionCLS = round(precision_score(y_test, y_pred), 4)
    # print(f1_score(y_test, y_pred))
    f1_Score = round(f1_score(y_test, y_pred), 4)
    # brier_score
    # predict probabilities
    probs = clf.predict_proba(X_test)
    probs = probs[:, 1]
    loss = round(brier_score_loss(y_test, probs), 4)

    # P_Value
    # x = [0.7721, 0.7691, 0.1702]
    # y = [0.7921, 0.7891, 0.2000]
    # x = [accuracy, recall, precision]
    # y = [accuracy1, recall2, precision3]
    # P_value = scipy.stats.ranksums(x, y)
    # print(P_value)

    # pandas dataframe to html table flask
    # uploaded_df_html = uploaded_df.to_html()
    # return render_template('show_csv_data.html', data_var=uploaded_df_html)
    return render_template('InterfaceDesign2.html', accuracy_score_before='Accuracy Score = {}'.format(accuracyCLS),
                           recall_score_before='Recall Score = {}'.format(recallCLS), precision_score_before='Precision Score = {}'.format(precisionCLS), f1_score_before='F1 Score = {}'.format(f1_Score),
                           barier_scoreCLS_before='Barier Score = {}'.format(loss))


# (2)----------------------------------------- Regression Model ----------------------------------------------#

@app.route('/show_result_regression', methods=["POST"])
def showResultreg():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import brier_score_loss

    # by convention our package is to be imported as dp (dp for Differential Privacy!)
    import numpy as np
    import matplotlib.pyplot as plt

    # X = uploaded_df.drop('Outcome', axis=1)
    # y = uploaded_df[['Outcome']]

    X = uploaded_df.drop('Outcome', axis=1)
    y = uploaded_df[['Outcome']]

    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    encoder = OrdinalEncoder()
    encoder.fit(X)
    X_encoded = encoder.transform(X)
    # ---------------------#

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.33, random_state=42)
    print(X_train.shape)  # type: ignore
    print(X_test.shape)   # type: ignore

    lr = LogisticRegression()
    model = lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    global r_squaredREG
    r_squaredREG = round(model.score(X_encoded, y), 4)
    print(r_squaredREG)

    global adj_r_squaredREG
    adj_r_squaredREG = round(1 - (1-model.score(X_encoded, y)) *
                             (len(y)-1)/(len(y)-X_encoded.shape[1]-1), 4)
    print(adj_r_squaredREG)

    # predict probabilities
    probs = model.predict_proba(X_test)
    # keep the predictions for class 1 only
    probs = probs[:, 1]
    # calculate bier score
    global lossREG
    lossREG = round(brier_score_loss(y_test, probs), 4)
    print(lossREG)

    # P_Value
    import scipy
    # x = [r_squared, adj_r_squared, loss]
    # y = [r_squared, adj_r_squared, loss]
    # P_value = scipy.stats.ranksums(x, y)
    # print(P_value)

    return render_template('InterfaceDesign2.html', r_squared_value_before='R_Squared Value = {}'.format(r_squaredREG),
                           adj_r_squared_value_before='Adjust R_Squared Value = {}'.format(adj_r_squaredREG), barier_scoreREG_before='Barier Score = {}'.format(lossREG))


# (3)----------------------------------------- Clustering Model ----------------------------------------------#

@app.route('/show_result_clustering', methods=["POST"])
def showResultclust():
    from sklearn.preprocessing import OrdinalEncoder
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)

    # libraries
    from sklearn.cluster import KMeans

    # dataset = pd.read_csv("Mall_Customers.csv")
    # df = pd.DataFrame(dataset)
    # print(df)

    # Missing values computation
    # null_value = dataset.isnull().sum()
    # print(null_value)
    # Feature sleection for the model
    # Considering only 2 features (Annual income and Spending Score) and no Label available
    X = uploaded_df.iloc[:, [3, 4]].values

    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    encoder = OrdinalEncoder()
    encoder.fit(X)
    X_encoded = encoder.transform(X)

    # to figure out K for KMeans, I will use ELBOW Method on KMEANS++ Calculation
    wcss = []

    # we always assume the max number of cluster would be 10
    # you can judge the number of clusters by doing averaging
    # Static code to get max no of clusters

    for i in range(1, 2):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(X_encoded)
        wcss.append(kmeans.inertia_)

        # inertia_ is the formula used to segregate the data points into cluster

    # Visualizing the ELBOW method to get the optimal value of K
    # plt.plot(range(1, 11), wcss)
    # plt.title('The Elbow Method')
    # plt.xlabel('no of clusters')
    # plt.ylabel('wcss')
    # plt.show()
    global InirtiaCLUST
    InirtiaCLUST = (str(wcss))

    # P_Value
    # import scipy
    # x = [Inirtia]
    # y = [Inirtia]
    # P_value = scipy.stats.ranksums(x, y)
    # print(P_value)
    return render_template('InterfaceDesign2.html', Inirtia_value_before='The Inirtia Value is = {}'.format(InirtiaCLUST))


# ------------------------------------After Di-identificvation---------------------------------------------------------------
# ------------------------------------After Di-identificvation---------------------------------------------------------------

@app.route('/show_result_classification_after', methods=["POST"])
def showResult2():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)

    # Importing the libraries
    # import pandas as pd

    # Import Decision Tree Classifier

    # Import train_test_split function

    # split dataset in features and target variable
    # feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'Email', 'DiabetesPedigreeFunction']

    # X = uploaded_df[feature_cols]  # Features
    # y = uploaded_df.Outcome  # Target variable

    X = uploaded_df.drop('Outcome', axis=1)
    y = uploaded_df[['Outcome']]

    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    encoder = OrdinalEncoder()
    encoder.fit(X)
    X_encoded = encoder.transform(X)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=1)  # 70% training and 30% test

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # print(y_pred)

    # print(accuracy)
    accuracy = round(accuracy_score(y_pred, y_test), 4)
    print(accuracy)

    # print('Recall: %.3f' % recall_score(y_test, y_pred))
    recall = round(recall_score(y_test, y_pred), 4)
    # print(('precision_score(y_test, y_pred)))
    precision = round(precision_score(y_test, y_pred), 4)
    # print(f1_score(y_test, y_pred))
    f1_Score = round(f1_score(y_test, y_pred), 4)
    # brier_score
    # predict probabilities
    probs = clf.predict_proba(X_test)
    probs = probs[:, 1]
    loss = round(brier_score_loss(y_test, probs), 4)

    # P_Value
    # x = [0.7721, 0.7691, 0.1702]
    # y = [0.7921, 0.7891, 0.2000]
    x = [accuracyCLS, recallCLS, precisionCLS]
    y = [accuracy, recall, precision]
    P_valueCLS = scipy.stats.ranksums(x, y)
    # print(P_valueCLS)

    # pandas dataframe to html table flask
    # uploaded_df_html = uploaded_df.to_html()
    # return render_template('show_csv_data.html', data_var=uploaded_df_html)
    return render_template('InterfaceDesign2.html', accuracy_score_after='Accuracy Score = {}'.format(accuracy),
                           recall_score_after='Recall Score = {}'.format(recall), precision_score_after='Precision Score = {}'.format(precision), f1_score_after='F1 Score = {}'.format(f1_Score),
                           barier_scoreCLS_after='Barier Score = {}'.format(loss), p_valueCLS='The P Value = {}'.format(P_valueCLS))


# (2)----------------------------------------- Regression Model ----------------------------------------------#

@app.route('/show_result_regression_after', methods=["POST"])
def showResultreg2():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import brier_score_loss

    # by convention our package is to be imported as dp (dp for Differential Privacy!)
    import numpy as np
    import matplotlib.pyplot as plt

    # X = uploaded_df.drop('Outcome', axis=1)
    # y = uploaded_df[['Outcome']]

    X = uploaded_df.drop('Outcome', axis=1)
    y = uploaded_df[['Outcome']]

    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    encoder = OrdinalEncoder()
    encoder.fit(X)
    X_encoded = encoder.transform(X)
    # ---------------------#

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.33, random_state=42)
    print(X_train.shape)  # type: ignore
    print(X_test.shape)   # type: ignore

    lr = LogisticRegression()
    model = lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    r_squared = round(model.score(X_encoded, y), 4)
    print(r_squared)

    adj_r_squared = round(1 - (1-model.score(X_encoded, y)) *
                          (len(y)-1)/(len(y)-X_encoded.shape[1]-1), 4)
    print(adj_r_squared)

    # predict probabilities
    probs = model.predict_proba(X_test)
    # keep the predictions for class 1 only
    probs = probs[:, 1]
    # calculate bier score
    loss = round(brier_score_loss(y_test, probs), 4)
    print(loss)

    # P_Value
    import scipy
    x = [r_squaredREG, adj_r_squaredREG, lossREG]
    y = [r_squared, adj_r_squared, loss]
    P_valueREG = scipy.stats.ranksums(x, y)
    print(P_valueREG)

    return render_template('InterfaceDesign2.html', r_squared_value_after='R_Squared Value = {}'.format(r_squared),
                           adj_r_squared_value_after='Adjust R_Squared Value = {}'.format(adj_r_squared), barier_scoreREG_after='Barier Score = {}'.format(loss), p_valueREG='The P Value = {}'.format(P_valueREG))


# (3)----------------------------------------- Clustering Model ----------------------------------------------#

@app.route('/show_result_clustering_after', methods=["POST"])
def showResultclust2():
    from sklearn.preprocessing import OrdinalEncoder
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)

    # libraries
    from sklearn.cluster import KMeans

    # dataset = pd.read_csv("Mall_Customers.csv")
    # df = pd.DataFrame(dataset)
    # print(df)

    # Missing values computation
    # null_value = dataset.isnull().sum()
    # print(null_value)
    # Feature sleection for the model
    # Considering only 2 features (Annual income and Spending Score) and no Label available
    X = uploaded_df.iloc[:, [3, 4]].values

    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    encoder = OrdinalEncoder()
    encoder.fit(X)
    X_encoded = encoder.transform(X)

    # to figure out K for KMeans, I will use ELBOW Method on KMEANS++ Calculation
    wcss = []

    # we always assume the max number of cluster would be 10
    # you can judge the number of clusters by doing averaging
    # Static code to get max no of clusters

    for i in range(1, 2):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(X_encoded)
        wcss.append(kmeans.inertia_)

        # inertia_ is the formula used to segregate the data points into cluster

    # Visualizing the ELBOW method to get the optimal value of K
    # plt.plot(range(1, 11), wcss)
    # plt.title('The Elbow Method')
    # plt.xlabel('no of clusters')
    # plt.ylabel('wcss')
    # plt.show()
    Inirtia = (str(wcss))

    # P_Value
    import scipy
    x = [InirtiaCLUST]
    y = [Inirtia]
    P_valueCLUST = scipy.stats.ranksums(x, y)
    # print(P_valueCLUST)
    return render_template('InterfaceDesign2.html', Inirtia_value_after='The Inirtia Value is = {}'.format(Inirtia), p_valueCLUST='The P Value = {}'.format(P_valueCLUST))


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='206.189.146.29')
