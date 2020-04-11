#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import nltk
import pickle
import re
import sys
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[ ]:


def load_data(database_filepath):
    """
    INPUT -> The database filepath where our database exists
    OUTPUT -> X, Y astype(pd.DataFrame), categories names astype(str)
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df.message
    Y = df.loc[:, ~df.columns.isin(['id', 'message', 'original', 'genre'])]
    col_names = list(df.columns[4:])
    return X, Y, col_names


# In[ ]:


def tokenize(text):
    """
    This function tokenize all the messages given to be ready to be fitted
    
    INPUT -> The text needs to be fitted astype(str)
    OUTPUT -> the clean data astype(list)
    """
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    tokens = word_tokenize(text)
    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words('english'):
            clean_tokens.append(tok)
    lemmatizer = WordNetLemmatizer()
    clean_nouns = [lemmatizer.lemmatize(tok).strip() for tok in clean_tokens]
    clean_data = [lemmatizer.lemmatize(tok, pos='v').strip() for tok in clean_nouns]
    return clean_data


# In[ ]:


def build_model():
    """
    Build our classification model
    INPUT -> It takes no input
    OUTPUT -> The classification model
    """
    pipeline = Pipeline([
    ('vec', CountVectorizer(tokenizer=tokenize)),
    ('tfid', TfidfTransformer()),
    ('clf', MultiOutputClassifier(MultinomialNB())),
    ])
    
    parameters = {
        'clf__estimator__alpha': [1, 0.25],
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


# In[ ]:


def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT -> Our model, Xtest dataframe, ytest dataframe astype(pd.DataFrame)
    OUTPUT -> Prints the accuracy rate of each category
    """
    y_pred = cv.predict(X_test)
    
    for i in range(len(category_names)):
        report = classification_report(y_test.iloc[:,i].values, y_pred[:,i])
        print(f"Accuracy of {category_names[i]}: \n", report)
    


# In[ ]:


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


# In[ ]:


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '              'as the first argument and the filepath of the pickle file to '              'save the model to as the second argument. \n\nExample: python '              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


# In[ ]:


if __name__ == '__main__':
    main()

