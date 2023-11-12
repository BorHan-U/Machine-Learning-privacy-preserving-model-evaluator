# Machine-Learning-privacy-preserving-model-evaluator

## ML-Model-Flask-Deployment
This is a demo project to elaborate on how Machine Learn Models are deployed on production using Flask API

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Learning Model), and Flask (for API) installed.
Also,

• Flask
• Sklearn
• Pandas
• ordinalEncoder
• os
• Flask (render template, request, session)
• werkzeug.utils

### Project Structure
This project has four major parts :
1. Flask_ML_Evaluator.py - This contains code for our Machine Learning models (classification, regression, clustering).
2. .py - This contains Flask APIs that receive data details through GUI or API calls, compute the precited value based on our model, and return it.
3. request.py - This uses the requests module to call APIs already defined in app.py and displays the returned value.
4. templates - This folder contains the HTML template to customer the design.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running the below command -
```
2. Run app.py using the below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage as below :
<img width="1066" alt="Screenshot 2023-11-12 at 8 38 06 PM" src="https://github.com/BorHan-U/Machine-Learning-privacy-preserving-model-evaluator/assets/55747898/c3487f0e-a688-48ba-aa0b-11c22279c11c">

upload datasets after and before de-identification and compare the result of your preferences.

If everything goes well, you should  be able to see the predicted values by three models on the HTML page!
for instance:
<img width="430" alt="Screenshot 2023-11-12 at 8 41 52 PM" src="https://github.com/BorHan-U/Machine-Learning-privacy-preserving-model-evaluator/assets/55747898/e1447944-20a6-4c2d-b012-23381ee90df7">

4. You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the below command to send the request with some pre-popuated values -
```
python request.py
``

**##Note for HTML**
After importing all libraries and packages, respective codes will be written following the available coding file. And, must render the HTML and CSS file to get the frontend design.

As there will be several files, we should run only the main App (HTML and CSS files already rendered into the app) file. Once the app is running on the terminal, it will give a result of active running file status with an address of the localhost server. Copy and Run the address on the web server and the model will be visible on the server with the functionality included at. Then, Upload & get the preference model result.

In addition, if CSS is improved in the future, the file must link with the HTML file, and render the HTML file in the main.py. So, the changes will be effected properly.

**#Evaluation:**
To evaluate the system, for every new dataset must alter the features column. This is the only thing must needs to be accounted in order to run the code successfully. Example code is below:

The column may varies from datasets to new datasets. Since it is a diabetes dataset, it has such as columns BMI, insulin, etc. However, to evaluate, I have used numerous datasets.

• Pima Indian diabetes datasets
• IoT temperatures datasets
• Fatal health classification datasets
• Mall customer datasets
• Smart Home Dataset with weather Information datasets
• Student Mental health datasets


# Machine-Learning-Evaluator-for-privacy-preserving
