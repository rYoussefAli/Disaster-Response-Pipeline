# Disaster Response Pipeline
Machine Learning project aims to classify the messeges in case of disaster
![](https://sites.uci.edu/emergencymanagement/files/2017/02/disastercollage.png)

# Table of Content
1. [Installation](#Installation)
2. [Initialize](#Initialize)
3. [Motivation](#Motivations)
4. [LICENSE](#LICENSE)

# Installation
Necessary libraries: <br>
You should  face no issue running the file using Anaconda3 distribution

# Initialize
1. Clone the full repository
2. Run the data/process_data.py file to create the clean database*
3. Run the models/train_classifier.py script to create the new model*
- Notice*: These Python scripts should be able to run with additional arguments 
specifying the files used for the data and model.
  - i.e: ```python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db```
  - i.e: ```python train_classifier.py ../data/DisasterResponse.db classifier.pkl``` <br> 
- Doing that you have made your trained model and you are ready to predict and visualize your data <br>

> To visualize your data: <br>
4. Run the app/run.py script
5. Open http://0.0.0.0:3001/

# Motivations
While disaster, you have no time to think where I should send my SOS message, so I made this project to predict based on real messages that were sent during past disaster events, the category of new messages. It is a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

# LICENSE
This project is Under [GNU LICENSE](https://github.com/YoussefAli99/Disaster-Response-Pipeline/blob/master/LICENSE)
