# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:35:59 2020

@author: mark
"""

import pandas as pd
from google.cloud import bigquery

SQL = """
    SELECT ContractId, RegistrationDate
    FROM `afcproj.files_dupe.Contracts_20201117` 
    WHERE RegistrationDate < DATETIME({endyear}, {endmonth}, 01, 00, 00, 00)  -- EOM + 1 DAY
    and RegistrationDate >= DATETIME({startyear}, {startmonth}, 01, 00, 00, 00) -- FIRST DAY OF MONTH
"""

project = 'afcproj'
dataset = 'files_dupe'

client = bigquery.Client()

date_range_start = '2017-06-01'
date_range_end = '2020-10-31'

drange = pd.date_range(date_range_start, date_range_end, freq='1MS')

for startmonth, endmonth in zip(drange, drange.shift(1)):
    
    view_name = '{}_{}'.format(startmonth.month_name(),startmonth.year,)
    view_id = "{project}.{dataset}.{viewname}".format(project=project, dataset=dataset, viewname=view_name)
    view = bigquery.Table(view_id)
    
    # The source table in this example is created from a CSV file in Google
    # Cloud Storage located at
    # `gs://cloud-samples-data/bigquery/us-states/us-states.csv`. It contains
    # 50 US states, while the view returns only those states with names
    # starting with the letter 'W'.
    view.view_query = SQL.format(endyear=endmonth.year, endmonth=endmonth.month, 
                                 startyear=startmonth.year, startmonth=startmonth.month)
    
    # Make an API request to create the view.
    view = client.create_table(view)
    print(f"Created {view.table_type}: {str(view.reference)}")