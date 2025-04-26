import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import letor_metrics
# Load the data
data_path = 'mt_updated.csv'
data = pd.read_csv(data_path)

linguistic_features = ['GENETIC', 'SYNTACTIC', 'FEATURAL', 'PHONOLOGICAL', 'INVENTORY', 'GEOGRAPHIC']
languages = data['Source lang'].unique()
data['GENETIC_relevance'] = 0
data['SYNTACTIC_relevance'] = 0
data['FEATURAL_relevance'] = 0
data['PHONOLOGICAL_relevance'] = 0
data['INVENTORY_relevance'] = 0
data['GEOGRAPHIC_relevance'] = 0

data['relevance'] = 0
# assign relevances for translation score
for source_lang in data['Source lang'].unique():
    source_lang_data = data[data['Source lang'] == source_lang].copy()
    source_lang_data['rank'] = source_lang_data['BLEU'].rank(method='min', ascending=False)
    top_indices = source_lang_data[source_lang_data['rank'] <= 3].index
    data.loc[top_indices, 'relevance'] = 4 - source_lang_data.loc[top_indices, 'rank']


# also assign relevances for each linguistic feature
for feature in linguistic_features:
    for source_lang in data['Source lang'].unique():
        source_lang_data = data[data['Source lang'] == source_lang].copy()
        source_lang_data['rank'] = source_lang_data[feature].rank(method='min', ascending=True)
        top_indices = source_lang_data[source_lang_data['rank'] <= 3].index
        data.loc[top_indices, f'{feature}_relevance'] = 4 - source_lang_data.loc[top_indices, 'rank']

results = {}

# compare the top 3 for each with ndcg@3
for feature in linguistic_features:
    ndcg_scores = []
    for lang in languages:

        lang_data = data[data['Source lang'] == lang].copy()
        true_relevance = lang_data['relevance']
        lan_relevance = lang_data[f'{feature}_relevance']
        ndcg = ndcg_score([true_relevance], [lan_relevance], k=3)
        # ndcg = letor_metrics.ndcg_score(true_relevance.values, lan_relevance.values,k=3)
        ndcg_scores.append(ndcg)

    results[feature] = np.mean(ndcg_scores)


for feature, mean_ndcg in results.items():
    print(f'Mean NDCG@3 for {feature}: {mean_ndcg*100}')

