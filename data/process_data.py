#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import our necessary libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


# In[2]:


def load_data(messages_filepath, categories_filepath):
    """
    This function serves to load our datasets to the pandas DataFrame
    
    INPUT -> messages and categories filepath astype(str)
    OUTPUT -> returns a merged DataFrame
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    return df


# In[3]:


def clean_data(df):
    """
    This function takes our dataframe that needs to be cleaned
    
    INPUT ->  astype(pd.DataFrame)
    OUTPUT -> The DataFrame cleaned
    """
    # create a dataframe of the n individual category columns
    categories = df['categories'].str.split(';', expand=True)
    row = next(categories.iterrows())[1]
    category_colnames = pd.Series(row).apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories.columns:
        # set each value to be the last character of the string and transfer it to numeric
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)
    
    return df


# In[4]:


def save_data(df, database_filename):
    """
    save_data function saves the dataframe in a sqlite database
    
    INPUT -> pd.DataFrame, database_filename astype(str)
    OUTPUT -> This function has no output
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', con=engine, index=False, if_exists='replace')


# In[5]:


def main():
    """
    This main function will execute the ETL pipeline:
    1- Extraact out data from datasets
    2- Transform our data to 'clean_data'
    3- Load the cleaned dataframe into a sqlite database 
        
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '              'datasets as the first and second argument respectively, as '              'well as the filepath of the database to save the cleaned data '              'to as the third argument. \n\nExample: python process_data.py '              'disaster_messages.csv disaster_categories.csv '              'DisasterResponse.db')


# In[6]:


if __name__ == '__main__':
    main()

