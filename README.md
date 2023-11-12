# Machine-Learning-privacy-preserving-model-evaluator

Guidelines for python machine learning privacy evaluator (FLASK)
–-To start the project the instruction should be followed and maintain the sequences as well –
1. Installation of Python
Python: To run the model, the required python version is 3.11. Here is the download process Weblink>> https://www.python.org/downloads/
• Select Version of Python to Install (python-3, version - 3.11.2)
• During installation, press “yes” whenever it appeared
• Verify Python Was Installed On the machine by entering ‘python 3’ command in console
• Verify Pip Was Installed (Enter pip -V in the console)
Upon a click python 3.11.2 will be install on selected path and place the file in a known path to maintain the same directory for all upcoming files.
2. Installation of Vscode (Visual Studio Code):
Any latest VScode will adaptable with the model and make sure the VScode is in a latest form. Also, keep the Vscode in same path where python is installed previously.
website>> https://code.visualstudio.com/download
   
 1. Download Visual studio code for macOS through given link
2. Open the browser's download list and locate the downloaded app or archive.
3. Drag Visual Studio Code to the Applications folder, making it available in the macOS Launchpad.
4. Open VS Code from the Applications folder, by double clicking the icon.
5. Add VS Code to your Dock by right-clicking on the icon, located in the Dock, to bring up the context
menu and choosing Options, Keep in Dock.
6. Launch VS Code.
7. Open the Command Palette (Cmd+Shift+P)/ terminal and type 'shell command' to find the Shell
Command: Install 'code' command in PATH command.
8. Restart the terminal for the new $PATH value to take effect. You'll be able to type 'code .' in any folder to start editing files in that folder.
Note: Upon a click on the preference VsCode (mac or windows) the software will be installed on the selected path. Maintain the same path as python installed.
3. Virtual environment set:
This is most important trick that a virtual environment should create before install any packages and import any libraries. To work with packages, the packages must be at same virtual environment where coding will be written. To create the virtual environment the following command should run in the terminal.
  #for macOS
 python3 -m venv .venv
 #for windows
 python -m venv .venv

Once virtual environment created, select the .venv from the bottom of the terminal and proceed to install packages and libraries on the selected virtual environment.
3.1. After virtual environment set, here are the list of extensions that need to be installed from extensions section (indicated with red box)
Extensions:
 4.
• • • • • •
Python
Pylint
Pip packages (pip3)
HTML CSS support
Remote - SSH
And necessary update it system requested any in bottom right notification
Libraries and Packages
Once step (2.1) is done and Python and Vscode are ready on the machine, open the vscode and install required packages using terminal and import libraries using vscode shell.
To install from terminal use the command ‘pip install [package name]’.
Note: all packages need to install from terminal. To do this, use the command on the terminal ‘pip install [package name]’.

• Flask
• Sklearn
• Pandas
• ordinalEncoder
• os
• Flask (render template, request, session)
• werkzeug.utils
• An Example of installing packages from terminal:
(venv) (base) borhan@Borhans-MacBook-Pro Deployment-flask-master % pip install Flask
• Libraries import: For flask app (import):
To store the model in public folder (pickle):
For the all of three model (cls,regression, clustering), the required packages are:
  from flask import Flask, render_template, request, session import pandas as pd
import os
from werkzeug.utils import secure_filename
   import pickle
   #Encoder & skleran for the all model
from sklearn.preprocessing import OrdinalEncoder from sklearn.model_selection import train_test_split
#model classification
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score from sklearn.metrics import f1_score
from sklearn.metrics import brier_score_loss from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
#model regression
from sklearn.linear_model import LogisticRegression from sklearn.metrics import mean_absolute_error from sklearn.metrics import brier_score_loss
#model clustering
from sklearn.cluster import KMeans
 
After imported all libraries and packages, respective code will be written following availabe coding file. And, must render the HTML and CSS file to get the frontend design.
As there will be several file, we should run only main App (html and css files already render into the app) file. Once app is running on terminal, it will give a result of active running file status with an address of localhost server. Copy and Run the address at web server and model will be visible at the server with the functionality included at. Then, Upload & a get preference model result.
In addition, if CSS is improved in future, the file must link with HTML file, and render the HTML file in the main.py. So, the changes will effect properly.
Evaluation:
To evaluate the system, for every new datasets must alter the features column. This is only thing must need be accounted in order to run the code successfully. Example code is below:
The column may varies from datasets to a new datasets. Since it is a diabetes datasets, it has such as columns BMI, insulin, etc. However, to evaluate, I have used numerous datasets.
• Pima Indian diabetes datasets
• IoT temperatures datasets
• Fatal health classification datasets
• Mall customers datasets
• Smart Home Dataset with weather Information datasets
• Student Mental health datasets
