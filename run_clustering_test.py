import pandas as pd
from clustering import run_clustering

df = pd.read_csv('cleaned_bank_credit_data.csv')
selected = ['region','population_group','bank_group','occupation_group','year','no_of_accounts']
res = run_clustering(df, selected, algorithm='KMeans', params={'n_clusters':4})
print('labels unique:', set(res['labels']))
print('silhouette:', res['metrics'])
