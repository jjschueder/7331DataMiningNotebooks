# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:11:14 2020

@author: jjschued
"""

from google.cloud import bigquery

# Imports the Google Cloud client library
from google.cloud import storage
from google.oauth2 import service_account
import os
import pandas as pd
import pandas_gbq
project = 'flash-ward-264717'
credentials = service_account.Credentials.from_service_account_file(r'C:/Users/jjschued/Documents/SMU/7331 Machine Learning/My Project 35341-372d7a58bfb3.json')

# Construct a BigQuery client object.
client = bigquery.Client(credentials=credentials, project=project)

begindate = '2012-01-01'
enddate = '2012-03-01'
  
query = "select * FROM `bigquery-public-data.iowa_liquor_sales.sales` where date > '" +begindate + "' and date <= '" + enddate + "' LIMIT 20000" 
    

df = client.query(query).to_dataframe()

   
# now let's a get a summary of the variables 
print (df.info())
# we can see that most of the data 
#  is saved as an integer or as a nominal object

# Notice that all of the data is stored as a non-null object
# That's not good. It means we need to change those data types
# in order to encode the variables properly. Right now Pandas
# thinks all of our variables are nominal!

import numpy as np
# replace '?' with -1, we will deal with missing values later
df = df.replace(to_replace='?',value=-1) 

# let's start by first changing the numeric values to be floats
continuous_features = ['state_bottle_cost', 'state_bottle_retail', 'sale_dollars', 'volume_sold_liters', 'volume_sold_gallons']

# and the oridnal values to be integers
ordinal_features = ['bottles_sold']

# we won't touch these variables, keep them as categorical
categorical_features = ['pack', 'bottle_volume_ml', 'store_number', 'store_name', 'address', 'city', 'zip_code',
                  'county_number', 'county', 'category', 'category_name', 'vendor_number',
                  'vendor_name', 'item_number', 'item_description'];

                        

# use the "astype" function to change the variable type
df[continuous_features] = df[continuous_features].astype(np.float64)
df[ordinal_features] = df[ordinal_features].astype(np.int64)

df.info() # now our data looks better!!


dfstats = df.describe() # will get summary of continuous or the nominals

# let's set those values to NaN, so that Pandas understand they are missing
df = df.replace(to_replace=-1,value=np.nan) # replace -1 with NaN (not a number)
print (df.info())
dfstats2 = df.describe() # scroll over to see the values

one_hot_df = pd.concat([pd.get_dummies(df[col],prefix=col) for col in categorical_features], axis=1)

one_hot_df.head()

#https://stackoverflow.com/questions/31029560/plotting-categorical-data-with-pandas-and-matplotlib/31029857
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, len(categorical_features))
for i, categorical_feature in enumerate(df[categorical_features]):
    df[categorical_feature].value_counts().plot("bar", ax=ax[i]).set_title(categorical_feature)
fig.show()


categorical_features_sub = ['pack', 'bottle_volume_ml', 'city',
                  'county',  'category_name',
                  'vendor_name', 'item_description']

categorical_features_sub = ['city', 'county']

fig, ax = plt.subplots(1, len(categorical_features_sub))
for i, categorical_feature in enumerate(df[categorical_features_sub]):
    df[categorical_feature].value_counts().plot("bar", ax=ax[i]).set_title(categorical_feature)
fig.show()

df.hist(by=['city'])
