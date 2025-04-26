import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import letor_metrics
# Load the data2
data_path = 'mt.csv'
data = pd.read_csv(data_path)

dataset_features = ['Overlap word-level','Overlap subword-level','Transfer over target size ratio', 'Transfer target TTR distance']
des_features = ['Overlap word-level','Overlap subword-level','Transfer over target size ratio']
languages = data['Source lang'].unique()
data['Overlap word-level_relevance'] = 0
data['Overlap subword-level_relevance'] = 0
data['Transfer over target size ratio_relevance'] = 0
data['Transfer target TTR distance_relevance'] = 0

data['relevance'] = 0
# assign relevances for translation score
for source_lang in data['Source lang'].unique():
    source_lang_data = data[data['Source lang'] == source_lang].copy()
    source_lang_data['rank'] = source_lang_data['BLEU'].rank(method='min', ascending=False)
    top_indices = source_lang_data[source_lang_data['rank'] <= 3].index
    data.loc[top_indices, 'relevance'] = 4 - source_lang_data.loc[top_indices, 'rank']



# also assign relevances for each linguistic feature
for feature in des_features:
    for source_lang in data['Source lang'].unique():
        source_lang_data = data[data['Source lang'] == source_lang].copy()
        source_lang_data['rank'] = source_lang_data[feature].rank(method='min', ascending=False)
        top_indices = source_lang_data[source_lang_data['rank'] <= 3].index
        data.loc[top_indices, f'{feature}_relevance'] = 4 - source_lang_data.loc[top_indices, 'rank']

for source_lang in data['Source lang'].unique():
    source_lang_data = data[data['Source lang'] == source_lang].copy()
    source_lang_data['rank'] = source_lang_data['Transfer target TTR distance'].rank(method='min', ascending=True)
    top_indices = source_lang_data[source_lang_data['rank'] <= 3].index
    feature = 'Transfer target TTR distance'
    data.loc[top_indices, f'{feature}_relevance'] = 4 - source_lang_data.loc[top_indices, 'rank']
#
# t = data[['Overlap word-level','Overlap subword-level','Transfer over target size ratio', 'Transfer target TTR distance',
# 'Overlap word-level_relevance','Overlap subword-level_relevance',
# 'Transfer over target size ratio_relevance','Transfer target TTR distance_relevance']]
results = {}

# compare the top 3 for each with ndcg@3
for feature in dataset_features:
    ndcg_scores = []
    for lang in languages:

        lang_data = data[data['Source lang'] == lang].copy()
        true_relevance = lang_data['relevance']
        lan_relevance = lang_data[f'{feature}_relevance']
        ndcg = ndcg_score([true_relevance], [lan_relevance], k=3)
        # ndcg = letor_metrics.ndcg_score(true_relevance.values, lan_relevance.values,3)
        ndcg_scores.append(ndcg)

    results[feature] = np.mean(ndcg_scores)


for feature, mean_ndcg in results.items():
    print(f'Mean NDCG@3 for {feature}: {mean_ndcg*100}')

