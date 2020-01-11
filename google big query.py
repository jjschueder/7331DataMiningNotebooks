# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:52:47 2020

@author: jjschued
"""

from google.cloud import bigquery

# Imports the Google Cloud client library
from google.cloud import storage

# Instantiates a client
storage_client = storage.Client()

# The name for the new bucket
bucket_name = "my-new-bucket"

# Creates the new bucket
bucket = storage_client.create_bucket(bucket_name)

print("Bucket {} created.".format(bucket.name))



accountproject = 'MSDS7331'
Email_address = 'msds7331@flash-ward-264717.iam.gserviceaccount.com'
KeyIDs = '372d7a58bfb3ea3461f712f774bdbd044bbf6acf'

# Construct a BigQuery client object.
client = bigquery.Client()

query = """
    SELECT name, SUM(number) as total_people
    FROM `bigquery-public-data.usa_names.usa_1910_2013`
    WHERE state = 'TX'
    GROUP BY name, state
    ORDER BY total_people DESC
    LIMIT 20
"""
query_job = client.query(query)  # Make an API request.

print("The query data:")
for row in query_job:
    # Row values can be accessed by field name or index.
    print("name={}, count={}".format(row[0], row["total_people"]))