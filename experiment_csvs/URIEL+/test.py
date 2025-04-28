import pandas as pd

data = pd.read_csv('mt_updated.csv')
s = ['ara','est','aze','fas','msa']
filtered_df = data[~data['Source lang'].isin(s) & ~data['Transfer lang'].isin(s)]
filtered_df.to_csv('mt_removed_ambiguous_updated.csv',index=False)
