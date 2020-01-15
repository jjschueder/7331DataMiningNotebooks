  
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:11:14 2020
@author: jjschued
"""

from google.cloud import bigquery
from google.oauth2 import service_account
# Imports the Google Cloud client library
from google.cloud import storage
import os
import pandas as pd

project = 'flash-ward-264717'
credentials = service_account.Credentials.from_service_account_file(r'C:/Users/jjschued/Documents/SMU/7331 Machine Learning/My Project 35341-372d7a58bfb3.json')
#C:\Users\jjsch\Downloads
# Construct a BigQuery client object.
client = bigquery.Client(credentials=credentials, project=project)

begindate = '2012-01-01'
enddate = '2012-03-01'
  
query = "select * FROM `bigquery-public-data.iowa_liquor_sales.sales` where date > '" +begindate + "' and date <= '" + enddate + "' LIMIT 20000" 
df = client.query(query).to_dataframe()
    
query2 = "SELECT * FROM `bigquery-public-data.census_bureau_acs.zip_codes_2017_5yr` -- where geo_id = '51012' LIMIT 1000"


censusdf = client.query(query2).to_dataframe()
censusdf['zip_code'] = censusdf['geo_id']

liquorandcensusdf = pd.merge(df, censusdf, how = 'left', on='zip_code')

df.describe()
# in each column of dataframe
uniqueValues = df.nunique()
   
# now let's a get a summary of the variables 
print (df.info())
# we can see that most of the data 
#  is saved as an integer or as a nominal object

# Notice that all of the data is stored as a non-null object
# That's not good. It means we need to change those data types
# in order to encode the variables properly. Right now Pandas
# thinks all of our variables are nominal!

#add featyres
#https://www.interviewqs.com/ddi_code_snippets/extract_month_year_pandas
df['year'] = (pd.to_datetime(df['date']).dt.to_period('Y'))
df['year'] = str(df['year'])
df['year'] = df['year'].str[9:13]

df['month'] =(pd.to_datetime(df['date']).dt.to_period('M'))
df['month'] = str(df['month'])
df['month'] = df['month'].str[9:16]
#get year

# get month

#get population

#get income
# convert to ranges

#poverty level



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
                  'vendor_name', 'item_number', 'item_description', 'year', 'month'];

                        

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





# modeled after : https://github.com/jakemdrew/EducationDataNC starting next line

uniqueValues = df.nunique()
   
# now let's a get a summary of the variables 
print (df.info())



df.info(verbose=True)


#Remove any fields that have the same value in all rows
UniqueValueCounts = df.nunique(dropna=False)
SingleValueCols = UniqueValueCounts[UniqueValueCounts == 1].index
df2 = df.drop(SingleValueCols, axis=1)

#Review dataset contents after drops
print('*********After: Removing columns with the same value in every row.*******************')
df2.info(verbose=False)
print ('\r\nColumns Deleted: ', len(SingleValueCols))

#Remove any fields that have unique values in every row
df2RecordCt = df2.shape[0]
UniqueValueCounts = df2.apply(pd.Series.nunique)
AllUniqueValueCols = UniqueValueCounts[UniqueValueCounts == df2RecordCt].index
df3 = df2.drop(AllUniqueValueCols, axis=1)

#Review dataset contents after drops
print('*********After: Removing columns with unique values in every row.*******************')
df3.info(verbose=False)
print ('\r\nColumns Deleted: ', len(AllUniqueValueCols))

#Remove any empty fields (null values in every row)
df3DataRecordCt = df3.shape[0]
NullValueCounts = df3.isnull().sum()
NullValueCols = NullValueCounts[NullValueCounts == df3DataRecordCt].index
df4 = df3.drop(NullValueCols, axis=1)

#Review dataset contents after empty field drops
print('*********After: Removing columns with null / blank values in every row.*************')
df4.info(verbose=False)
print ('\r\nColumns Deleted: ', len(NullValueCols))


#Isolate continuous and categorical data types
#These are indexers into the schoolData dataframe and may be used similar to the schoolData dataframe 
D_boolean = df4.loc[:, (df4.dtypes == bool) ]
D_nominal = df4.loc[:, (df4.dtypes == object)]
D_continuous = df4.loc[:, (df4.dtypes != bool) & (df4.dtypes != object)]
print ("Boolean Columns: ", D_boolean.shape[1])
print ("Nominal Columns: ", D_nominal.shape[1])
print ("Continuous Columns: ", D_continuous.shape[1])
print ("Columns Accounted for: ", D_nominal.shape[1] + D_continuous.shape[1] + D_boolean.shape[1])

#Missing Data Threshold (Per Column)
missingThreshold = 0.60


#Eliminate continuous columns with more than missingThreshold percentage of missing value
DataRecordCt = D_continuous.shape[0]
missingValueLimit = DataRecordCt * missingThreshold
NullValueCounts = D_continuous.isnull().sum()
NullValueCols = NullValueCounts[NullValueCounts >= missingValueLimit].index
df5 = df4.drop(NullValueCols, axis=1)

#Review dataset contents after empty field drops
print('*********After: Removing columns with >= missingThreshold % of missing values******')
df5.info(verbose=False)
print ('\r\nColumns Deleted: ', len(NullValueCols))


#Unique Value Threshold (Per Column)
#Delete Columns >  uniqueThreshold unique values prior to one-hot encoding. 
#(each unique value becomes a new column during one-hot encoding)
uniqueThreshold = 20
df5.nunique()
#Delete categorical columns with > 100 unique values (Each unique value becomes a column during one-hot encoding)
oneHotUniqueValueCounts = df5[D_nominal.columns].apply(lambda x: x.nunique())
oneHotUniqueValueCols = oneHotUniqueValueCounts[oneHotUniqueValueCounts >= uniqueThreshold].index
df5.drop(oneHotUniqueValueCols, axis=1, inplace=True) 

#Review dataset contents one hot high unique value drops
print('*********After: Removing columns with >= uniqueThreshold unique values***********')
df5.info(verbose=False)
print ('\r\nColumns Deleted: ', len(oneHotUniqueValueCols))

#Isolate remaining categorical variables
begColumnCt = len(df5.columns)
D_nominal = df5.loc[:, (df5.dtypes == object or df5.dtypes == period[M])]

#one hot encode categorical variables
df6 = pd.get_dummies(data=df5, columns=D_nominal, drop_first=True)

categorical_features = ['pack', 'bottle_volume_ml', 'city',
                  'county',  'category_name',
                  'vendor_name', 'item_description']
one_hot_df6 = pd.concat([pd.get_dummies(df5[col],prefix=col) for col in D_nominal.columns], axis=1)

one_hot_df.head()

#Determine change in column count
endColumnCt = len(df6.columns)
columnsAdded = endColumnCt - begColumnCt

#Review dataset contents one hot high unique value drops
print ('Columns To One-Hot Encode: ', len(D_nominal.columns))
print('\r\n*********After: Adding New Columns Via One-Hot Encoding*************************')
df6.info(verbose=False)
print ('\r\nNew Columns Created Via One-Hot Encoding: ', columnsAdded)

# this python magics will allow plot to be embedded into the notebook
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
%matplotlib inline

# lets look at the boxplots separately
vars_to_plot_separate = [['pack','state_bottle_cost', 'state_bottle_retail'],
                         ['bottle_volume_ml','sale_dollars'],
                         ['bottles_sold', 'volume_sold_liters']]

plt.figure(figsize=(15, 7))

for index, plot_vars in enumerate(vars_to_plot_separate):
    plt.subplot(len(vars_to_plot_separate)/2, 
                3, 
                index+1)
    ax = df5.boxplot(column=plot_vars)
    
plt.show()

from pandas.plotting import scatter_matrix

from pandas.plotting import scatter_matrix
ax = scatter_matrix(df5, figsize=(15, 10))

countycounts = df['county'].value_counts()
countycountsdf = countycounts.to_frame()

survival_counts = pd.crosstab([df_imputed['Pclass'],df_imputed['Sex']], 
                              df_imputed.Survived.astype(bool))
countycountsdf.plot(kind='bar', 
                     stacked=True)

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

x = countycountsdf.index
energy = countycountsdf.county

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, energy, color='green')
plt.xlabel("County")
plt.ylabel("# Transactions")
plt.title("Number of Transactions by County")

plt.xticks(x_pos, x)
plt.figure(figsize=(20, 20))
plt.show()
