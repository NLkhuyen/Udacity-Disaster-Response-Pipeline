# Disaster Response Pipeline Project

## Table of Contents
1. [Project Description](#projectDescription)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [File Descriptions](#fileDescriptions)
	3. [Instructions](#instructions)

<a name="projectDescription"></a>
## Project Description
In the Project Workspace, we'll find a data set containing real messages that were sent during disaster events. Will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

Our project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly

<a name="fileDescriptions"></a>
### File Descriptions
1. App folder including the templates folder and "run.py" for the web application
2. Data folder containing "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data cleaning and transfering.
3. Models folder including "classifier.pkl" and "train_classifier.py" for the Machine Learning model.
4. README file
5. Preparation folder containing 6 different files, which were used for the project building. (Please note: this folder is not necessary for this project to run.)

<a name="instructions"></a>
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Etl_Disastor.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Etl_Disastor.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

<a name="githubRepo"></a>
### Github Repo

* [Udacity-Disaster-Response-Pipeline](https://github.com/NLkhuyen/Udacity-Disaster-Response-Pipeline)