  
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:11:14 2020
@author: jjschued
"""

from google.cloud import bigquery
from google.oauth2 import service_account
# Imports the Google Cloud client library
#from google.cloud import storage
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

#get unique list of categories
liquorcatlist = df['category_name'].unique().tolist()
liquortcatlistdf = pd.DataFrame(liquorcatlist)
#download to categorize in more summarized manner
#liquortcatlistdf.to_csv("C:\\Users\\jjsch\\downloads\\liquorcats.csv")    
#import
liquortcatlistdf = pd.read_csv("C:\\Users\\jjsch\\downloads\\liquorcats.csv")

df = pd.merge(df, liquortcatlistdf, how = 'left', on='category_name')

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
#                  'county_number', 'county', 'category', 'category_name', 'vendor_number',
#                 'vendor_name', 'item_number', 'item_description', 'storeparent'];
                        
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

#convert bottle size and package size to categorical
df['pack'] = pd.Categorical(df.pack)
df['bottle_volume_ml'] = pd.Categorical(df.bottle_volume_ml)

#Remove any fields that have unique values in every row
dfRecordCt = df.shape[0]
UniqueValueCounts = df.apply(pd.Series.nunique)
AllUniqueValueCols = UniqueValueCounts[UniqueValueCounts == dfRecordCt].index
df = df.drop(AllUniqueValueCols, axis=1)

#summarizd stores
#df['storeparent'] = df['store_name'].str[:7]

# modeled after : https://github.com/jakemdrew/EducationDataNC starting next line

uniqueValues = df.nunique()
   
# now let's a get a summary of the variables 
print (df.info())
df.info(verbose=True)


####
#start merging with demographic data
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
bins = [0, 18, 30, 40, 50, 65, 100]
labels = ["child","adult20", "adult30", "adult40", "adult50", "senior"]
censusdf['medagerange'] = pd.cut(censusdf['median_age'], bins=bins, labels=labels)
censusdf.medagerange.describe()

#poulation grouping
bins = [0, 200, 1000, 5000, 10000, 50000, 100000, 1000000]
labels = ["village","townlt1K", "towns5K", "town10K", "city50K", "city100K", "citygt100k"]
censusdf['total_pop_ranges'] = pd.cut(censusdf['total_pop'], bins=bins, labels=labels)
censusdf.total_pop_ranges.describe()
#group median and income per captica

#final census dataframe with subset of columns and groupings(limit furhter of subset columns
finalcolumnlist =  ['total_pop_ranges', 'medagerange', 'educationattainment', 'emplomenthealth', 'largestincomegroup', 'total_pop', 'median_income', 'majorityrace', 'majoritysex', 'income_per_capita', 'housing_units', 'income_less_10000', 'income_10000_14999', 'income_15000_19999', 'income_20000_24999', 'income_25000_29999', 'income_30000_34999', 'income_35000_39999', 'income_40000_44999', 'income_45000_49999', 'income_50000_59999', 'income_60000_74999', 'income_75000_99999', 'income_100000_124999', 'income_125000_149999', 'income_150000_199999', 'income_200000_or_more', 'armed_forces', 'civilian_labor_force', 'employed_pop', 'unemployed_pop', 'not_in_labor_force', 'employed_agriculture_forestry_fishing_hunting_mining', 'employed_arts_entertainment_recreation_accommodation_food', 'employed_construction', 'employed_education_health_social', 'employed_finance_insurance_real_estate', 'employed_information', 'employed_manufacturing', 'employed_other_services_not_public_admin', 'employed_public_administration', 'employed_retail_trade', 'employed_science_management_admin_waste', 'employed_transportation_warehousing_utilities', 'employed_wholesale_trade', 'female_female_households', 'four_more_cars', 'gini_index', 'graduate_professional_degree', 'group_quarters', 'high_school_including_ged', 'households_public_asst_or_food_stamps', 'poverty', 'associates_degree', 'bachelors_degree', 'high_school_diploma', 'less_one_year_college', 'masters_degree', 'one_year_more_college', 'zip_code']
censusdf = censusdf[finalcolumnlist]
#merge demogrpahics with liquor data 
liquorandcensusdf = pd.merge(df, censusdf, how = 'left', on='zip_code')

#fix  blank zip codes data so join works
dfzip = pd.isnull(liquorandcensusdf["total_pop"]) 
dfzipdf = liquorandcensusdf[dfzip]
unqiquezips = dfzipdf ['zip_code'].unique().tolist()

#mapping - missing zips based on google searches of addresses
zipmap = [['50300', '50312'], ['712-2', ' 51529'],['50015', ' 50014'],['52004', ' 52001'],['52733', ' 52732'],['52084', ' 52804']]  
# Create the pandas DataFrame 
dfzipmap = pd.DataFrame(zipmap, columns = ['zipcode', 'tocode']) 
#add new zips back to df
# this didn't work! 
df['zip_code'] = df['zip_code'].map(dfzipmap.set_index('zipcode')['tocode'])

#join again
liquorandcensusdf = pd.merge(df, censusdf, how = 'left', on='zip_code')

#Unique Value Threshold (Per Column)
#Delete Columns >  uniqueThreshold unique values prior to one-hot encoding. 
#(each unique value becomes a new column during one-hot encoding)
uniqueThreshold = 100
ldunique = liquorandcensusdf.nunique()
#Delete categorical columns with > 100 unique values (Each unique value becomes a column during one-hot encoding)
oneHotUniqueValueCounts = liquorandcensusdf[D_nominal.columns].apply(lambda x: x.nunique())
oneHotUniqueValueCols = oneHotUniqueValueCounts[oneHotUniqueValueCounts >= uniqueThreshold].index
liquorandcensusdf.drop(oneHotUniqueValueCols, axis=1, inplace=True) 

#add other census fields to drop.
liquorandcensusdf = liquorandcensusdf.drop(columns=['county_number', 'category', 'vendor_number'])

#fix these still

#drop store, drop address, drop county number, drop county
#drop item number, category

#possible drops store name, city, store location, vendors


#Remove any fields that have the same value in all rows
UniqueValueCounts = liquorandcensusdf.nunique(dropna=False)
SingleValueCols = UniqueValueCounts[UniqueValueCounts == 1].index
liquorandcensusdf = liquorandcensusdf.drop(SingleValueCols, axis=1)

#Review dataset contents after drops
print('*********After: Removing columns with the same value in every row.*******************')
liquorandcensusdf.info(verbose=False)
liquorandcensusdf.info()
print ('\r\nColumns Deleted: ', len(SingleValueCols))


#Review dataset contents after drops
print('*********After: Removing columns with unique values in every row.*******************')
liquorandcensusdf.info(verbose=False)
print ('\r\nColumns Deleted: ', len(AllUniqueValueCols))

#Remove any empty fields (null values in every row)
liquorandcensusdfDataRecordCt = liquorandcensusdf.shape[0]
NullValueCounts = liquorandcensusdf.isnull().sum()
NullValueCols = NullValueCounts[NullValueCounts == df3DataRecordCt].index
liquorandcensusdf = liquorandcensusdf.drop(NullValueCols, axis=1)

#Review dataset contents after empty field drops
print('*********After: Removing columns with null / blank values in every row.*************')
liquorandcensusdf.info(verbose=False)
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
ax = scatter_matrix(df5,figsize=(10, 10))

df_grouped = df5.groupby(by=['vendor_name'])
print (df_grouped.describe())



# this python magics will allow plot to be embedded into the notebook
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
%matplotlib inline

# lets look at the boxplots separately
vars_to_plot_separate = [['state_bottle_cost', 'state_bottle_retail'],
                        ['sale_dollars'],
                        ['bottles_sold', 'volume_sold_liters']]

plt.figure(figsize=(20,10))

for index, plot_vars in enumerate(vars_to_plot_separate):
    plt.subplot(len(vars_to_plot_separate)/2, 
                3, 
                index+1)
    ax = liquorandcensusdf.boxplot(column=plot_vars)
    
plt.show()

# Start by just plotting what we previsously grouped!
plt.style.use('ggplot')
plt.figure(figsize=(20,10))
df_grouped = liquorandcensusdf.groupby(by=['Category'])
sales_rate = df_grouped.sale_dollars.sum()
ax = sales_rate.plot(kind='barh')


ax = liquorandcensusdf.boxplot(column='state_bottle_cost', by = 'Category')
ax.set_yscale('log')

df_grouped = liquorandcensusdf.groupby(by=['bottle_volume_ml'])
sales_rate2 = df_grouped.sale_dollars.sum()
ax = sales_rate2.plot(kind='barh')


df_grouped = liquorandcensusdf.groupby(by=['total_pop_ranges'])
sales_rate3 = df_grouped.sale_dollars.sum()
ax = sales_rate3.plot(kind='barh')


df_grouped = liquorandcensusdf.groupby(by=['medagerange'])
sales_rate4 = df_grouped.sale_dollars.sum()
ax = sales_rate4.plot(kind='barh')

df_grouped = liquorandcensusdf.groupby(by=['educationattainment'])
sales_rate5 = df_grouped.sale_dollars.sum()
ax = sales_rate5.plot(kind='barh')

df_grouped = liquorandcensusdf.groupby(by=['largestincomegroup'])
sales_rate4 = df_grouped.sale_dollars.sum()
ax = sales_rate4.plot(kind='barh')

liquorandcensusdf['transcounter'] = 1
# cross tabulate example from http://nbviewer.ipython.org/gist/fonnesbeck/5850463# 
df_counts = pd.crosstab([liquorandcensusdf['Category']], 
                              liquorandcensusdf.transcounter.sum())
df_counts.plot(kind='bar', 
                     stacked=True)

from pandas.plotting import scatter_matrix
ax = scatter_matrix(liquorandcensusdf,figsize=(15, 10))