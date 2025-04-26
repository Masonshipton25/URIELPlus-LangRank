import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import letor_metrics
# Load the data
data_path = 'pos.csv'
data = pd.read_csv(data_path)

dataset_features = ['Overlap word-level','Transfer over target size ratio', 'Transfer target TTR distance']
des_features = ['Overlap word-level','Transfer over target size ratio']
languages = data['Task lang'].unique()
data['Overlap word-level_relevance'] = 0
data['Transfer over target size ratio_relevance'] = 0
data['Transfer target TTR distance_relevance'] = 0

data['relevance'] = 0
# assign relevances for translation score
for source_lang in data['Task lang'].unique():
    source_lang_data = data[data['Task lang'] == source_lang].copy()
    source_lang_data['rank'] = source_lang_data['Accuracy'].rank(method='min', ascending=False)
    top_indices = source_lang_data[source_lang_data['rank'] <= 3].index
    data.loc[top_indices, 'relevance'] = 4 - source_lang_data.loc[top_indices, 'rank']


def ndcg(rel_true, rel_pred):
    int_rel = rel_true.astype(int)
    rel_index = np.argsort(rel_pred)[::-1]
    true_indexed = int_rel[rel_index]
    int_rel[::-1].sort()
    dcg = (2 ** true_indexed[0] - 1) / np.log2(2) + (2 ** true_indexed[1] - 1) / np.log2(3) + (
                2 ** true_indexed[2] - 1) / np.log2(4)
    idcg = (2 ** int_rel[0] - 1) / np.log2(2) + (2 ** int_rel[1] - 1) / np.log2(3) + (2 ** int_rel[2] - 1) / np.log2(4)
    return dcg / idcg

# also assign relevances for each linguistic feature
for feature in des_features:
    for source_lang in data['Task lang'].unique():
        source_lang_data = data[data['Task lang'] == source_lang].copy()
        source_lang_data['rank'] = source_lang_data[feature].rank(method='min', ascending=False)
        top_indices = source_lang_data[source_lang_data['rank'] <= 3].index
        data.loc[top_indices, f'{feature}_relevance'] = 4 - source_lang_data.loc[top_indices, 'rank']

for source_lang in data['Task lang'].unique():
    source_lang_data = data[data['Task lang'] == source_lang].copy()
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

        lang_data = data[data['Task lang'] == lang].copy()
        true_relevance = lang_data['relevance']
        lan_relevance = lang_data[f'{feature}_relevance']
        score = ndcg_score([true_relevance], [lan_relevance],k=3)
        # score = letor_metrics.ndcg_score(true_relevance.values, lan_relevance.values,3)
        ndcg_scores.append(score)

    results[feature] = np.mean(ndcg_scores)


for feature, mean_ndcg in results.items():
    print(f'Mean NDCG@3 for {feature}: {round(mean_ndcg*100,1)}')

