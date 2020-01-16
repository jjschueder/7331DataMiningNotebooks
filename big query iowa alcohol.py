  
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
import numpy as np

project = 'flash-ward-264717'
credentials = service_account.Credentials.from_service_account_file(r'C:/Users/jjsch/Downloads/My Project 35341-372d7a58bfb3.json')
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

#which sex has more people
censusdfcolumns = list(censusdf.columns)
censussubsetcolumns = censusdfcolumns[14:30] + censusdfcolumns[55:70] +  censusdfcolumns[90:135] + censusdfcolumns[148:150] + censusdfcolumns[188:191] + censusdfcolumns[195:196] + censusdfcolumns[229:234]

censusdf['majoritysex'] = censusdf[['male_pop', 'female_pop']].idxmax(axis=1)

#which race is the majority
censusracecolumns = censusdfcolumns[18:25]
censusdf['majorityrace'] =  censusdf[censusracecolumns].idxmax(axis=1)
liquorandcensusdf = pd.merge(df, censusdf, how = 'left', on='zip_code')

#what income group has the most people
censusracecolumns = censusdfcolumns[55:70]
censusdf['largestincomegroup'] =  censusdf[censusracecolumns].idxmax(axis=1)

#are most employed or unemployed
censusemploycolumns = censusdfcolumns[148:150]
censusdf['emplomenthealth'] =  censusdf[censusemploycolumns].idxmax(axis=1)

#largest level of educational attainment
censusemploycolumns = censusdfcolumns[229:234]
censusdf['educationattainment'] =  censusdf[censusemploycolumns].idxmax(axis=1)


#medain age grouping

#poulation grouping

#group median and income per captica

#final census dataframe with subset of columns and groupings(limit furhter of subset columns)

#merge demogrpahics with liquor data 
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

import numpy as np
# replace '?' with -1, we will deal with missing values later
df = df.replace(to_replace='?',value=-1) 

# let's start by first changing the numeric values to be floats
continuous_features = ['state_bottle_cost', 'state_bottle_retail', 'sale_dollars', 'volume_sold_liters', 'volume_sold_gallons']

# and the oridnal values to be integers
ordinal_features = ['bottles_sold']

# we won't touch these variables, keep them as ca2'store_number', 'store_name', 'address', 'city', 'zip_code',
                  'county_number', 'county', 'category', 'category_name', 'vendor_number',
                  'vendor_name', 'item_number', 'item_description', 'storeparent'];

                        

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

#add featyres
df['year'] = df['date']
#get year

# get month
df['date1'] = to_datetime(df['date'])
#get population

#get income
# convert to ranges

#poverty level

#summarizd stores
df['storeparent'] = df['store_name'].str[:7]

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
uniqueThreshold = 100
df5.nunique()
#Delete categorical columns with > 100 unique values (Each unique value becomes a column during one-hot encoding)
oneHotUniqueValueCounts = df5[D_nominal.columns].apply(lambda x: x.nunique())
oneHotUniqueValueCols = oneHotUniqueValueCounts[oneHotUniqueValueCounts >= uniqueThreshold].index
df5.drop(oneHotUniqueValueCols, axis=1, inplace=True) 
df5 = df5.drop(columns=['county_number', 'category', 'vendor_number'])
#Isolate continuous and categorical data types
#These are indexers into the schoolData dataframe and may be used similar to the schoolData dataframe 
D_boolean = df5.loc[:, (df4.dtypes == bool) ]
D_nominal = df5.loc[:, (df4.dtypes == object)]
D_continuous = df5.loc[:, (df4.dtypes != bool) & (df4.dtypes != object)]
print ("Boolean Columns: ", D_boolean.shape[1])
print ("Nominal Columns: ", D_nominal.shape[1])
print ("Continuous Columns: ", D_continuous.shape[1])
print ("Columns Accounted for: ", D_nominal.shape[1] + D_continuous.shape[1] + D_boolean.shape[1])


one_hot_df6 = pd.concat([pd.get_dummies(df5[col],prefix=col) for col in D_nominal.columns], axis=1)

#Review dataset contents one hot high unique value drops
print('*********After: Removing columns with >= uniqueThreshold unique values***********')
one_hot_df6.info(verbose=False)
print ('\r\nColumns Deleted: ', len(oneHotUniqueValueCols))

# calculate the correlation matrix
corr_matrix  = one_hot_df6.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]



#Get all of the correlation values > 95%
x = np.where(upper > 0.95)

#Display all field combinations with > 95% correlation
cf = pd.DataFrame()
cf['Field1'] = upper.columns[x[1]]
cf['Field2'] = upper.index[x[0]]

#Get the correlation values for every field combination. (There must be a more pythonic way to do this!)
corr = [0] * len(cf)
for i in range(0, len(cf)):
    corr[i] =  upper[cf['Field1'][i]][cf['Field2'][i]] 
    
cf['Correlation'] = corr

print ('There are ', str(len(cf['Field1'])), ' field correlations > 95%.')
cf

#Check columns before drop 
print('\r\n*********Before: Dropping Highly Correlated Fields*************************************')
one_hot_df6.info(verbose=False)

# Drop the highly correlated features from our training data 
one_hot_df6 = one_hot_df6.drop(to_drop, axis=1)

#Check columns after drop 
print('\r\n*********After: Dropping Highly Correlated Fields**************************************')
one_hot_df6.info(verbose=False)

onehots_stats = one_hot_df6.describe()

#join one hot with df

mergedf = pd.merge(one_hot_df6, df5, left_index=True, right_index=True)

#scatter plot of all the numerics
from pandas.plotting import scatter_matrix
ax = scatter_matrix(df5,figsize=(15, 10))

df_grouped = df5.groupby(by=['vendor_name'])
print (df_grouped.describe())
