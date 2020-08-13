# Capstone Project
*Udacity - Machine Learning Engineer Nanodegree Program*

## Project Overview

In this project, our objective is to create a machine learning model which is capable of classifying a tweet into either a tweet related to a disaster or not. We analyse, clean and process the tweets (natural language) and build models to serve our purpose. As the problem is part of a Kaggle Competition, data is provided by Kaggle. Exploratory Data Analysis is carried out and important inferences are drawn to further drive our problem solving. The state of art NLP models such as BERT and TF-IDF are used. The results are compared for a number of variations in models and their hyperparameters. Tuning is done accordingly to come to the best results possible. In the end the best results are submitted as a part of Kaggle Competition Submission.

## Kaggle Competition Details
url: https://www.kaggle.com/c/nlp-getting-started/

## Software and Libraries

This project uses Python 3 and is designed to be completed through the Jupyter Notebooks IDE or any other program which lets you edit .ipynb files.
It is highly recommended  that you use the Anaconda distribution to install Python, since the distribution includes all necessary Python libraries
as well as Jupyter Notebooks.
It is recommended that you use Google Colab for implementing the BERT model as it provides free GPU, which is essential for accelerating the training process.

The following libraries are expected to be used in this project:

* NumPy
* pandas
* Sklearn / scikit-learn
* Matplotlib (for data visualization)
* Seaborn (for data visualization)
* Bert
* Tensorflow, Keras
* tensorflow_hub
* Keras: Layers :: Input, Dropout, GlobalAveragePooling1D
* Keras: Modesl :: Model, Sequential
* Keras : Callbacks :: ModelCheckpoint, EarlyStopping, Callback
* tokenization
* wordcloud.STOPWORDS
* collections.defaultdict
* GridSearchCV
* cross_val_score
* TfidfVectorizer
* CountVectorizer
* time
* re
* KNeighborsClassifier
* RandomForestClassifier
* SVC
* confusion_matrix
* accuracy_score
* train_test_split


## How the project is organized

There are 2 Jupyter Notebooks that are supposed to be ran in order

1. tweetClassification.ipynb
2. tweetClassification2.ipynb

The notebooks expect that the following files are present in the `data` folder:
1. train.csv
2. test.csv
3. sample_submission.csv

**The data is the provided by Kaggle, as the problem statement is a part of a Kaggle Competition**

## Additional data required by the project
1. Glove Embeddings: https://www.kaggle.com/authman/pickled-glove840b300d-for-10sec-loading
2. FastText Embeddings: https://www.kaggle.com/authman/pickled-crawl-300d-2m

