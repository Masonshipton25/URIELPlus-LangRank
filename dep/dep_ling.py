import pandas as pd
import numpy as np
import letor_metrics
from sklearn.metrics import ndcg_score
# Load the data
data_path = 'dep_updated.csv'
data = pd.read_csv(data_path)

linguistic_features = ['GENETIC', 'SYNTACTIC', 'FEATURAL', 'PHONOLOGICAL', 'INVENTORY', 'GEOGRAPHIC']
languages = data['Target lang'].unique()
data['GENETIC_relevance'] = 0
data['SYNTACTIC_relevance'] = 0
data['FEATURAL_relevance'] = 0
data['PHONOLOGICAL_relevance'] = 0
data['INVENTORY_relevance'] = 0
data['GEOGRAPHIC_relevance'] = 0

data['relevance'] = 0

def ndcg(rel_true, rel_pred):
    # to int array
    int_rel = rel_true.astype(int)

    # sort the ground truth array by predicted relevance (for dcg)
    rel_index = np.argsort(rel_pred)[::-1]
    true_indexed = int_rel[rel_index]

    # to find idcg, sort ground truth array descending
    int_rel[::-1].sort()

    # the top three languages given by the ranker are then used to calculate dcg using the relevance given by NLP model
    dcg = (2**true_indexed[0] - 1)/np.log2(2) + (2**true_indexed[1] - 1)/np.log2(3) + (2**true_indexed[2] - 1)/np.log2(4)
    # idcg always the same if max relevance is 10
    idcg = (2**int_rel[0] - 1)/np.log2(2) + (2**int_rel[1] - 1)/np.log2(3) + (2**int_rel[2] - 1)/np.log2(4)
    return dcg / idcg


# assign relevances for translation score
for source_lang in data['Target lang'].unique():
    source_lang_data = data[data['Target lang'] == source_lang].copy()
    source_lang_data['rank'] = source_lang_data['Accuracy'].rank(method='min', ascending=False)
    top_indices = source_lang_data[source_lang_data['rank'] <= 3].index
    data.loc[top_indices, 'relevance'] = 4 - source_lang_data.loc[top_indices, 'rank']


# also assign relevances for each linguistic feature
for feature in linguistic_features:
    for source_lang in data['Target lang'].unique():
        source_lang_data = data[data['Target lang'] == source_lang].copy()
        source_lang_data['rank'] = source_lang_data[feature].rank(method='min', ascending=True)
        top_indices = source_lang_data[source_lang_data['rank'] <= 3].index
        data.loc[top_indices, f'{feature}_relevance'] = 4 - source_lang_data.loc[top_indices, 'rank']


results = {}

# compare the top 3 for each with ndcg@3
for feature in linguistic_features:
    ndcg_scores = []
    for lang in languages:

        lang_data = data[data['Target lang'] == lang].copy()
        true_relevance = lang_data['relevance']
        lan_relevance = lang_data[f'{feature}_relevance']
        score = ndcg_score([true_relevance], [lan_relevance],k=3)
        # score = letor_metrics.ndcg_score(true_relevance.values, lan_relevance.values, 3)
        ndcg_scores.append(score)

    results[feature] = np.mean(ndcg_scores)


for feature, mean_ndcg in results.items():
    print(f'Mean NDCG@3 for {feature}: {round(mean_ndcg*100,1)}')


