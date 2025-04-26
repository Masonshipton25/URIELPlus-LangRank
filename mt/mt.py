import lightgbm
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRanker
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import letor_metrics


data = pd.read_csv('mt.csv')
# updates = pd.read_csv('mt_updated.csv')
# features = [ 'GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']
# for feature in features:
#     data[feature] = updates[feature]
groups = data['Source lang']
logo = LeaveOneGroupOut()
# Experiments with ALL, DATASET, and URIEL, respectively
features = [
    'Overlap word-level', 'Overlap subword-level', 'Transfer lang dataset size',
    'Target lang dataset size', 'Transfer over target size ratio', 'Transfer lang TTR',
    'Target lang TTR', 'Transfer target TTR distance', 'GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC'
]
# features = ['Overlap word-level', 'Overlap subword-level', 'Transfer lang dataset size',
#             'Target lang dataset size', 'Transfer over target size ratio', 'Transfer lang TTR',
#             'Target lang TTR', 'Transfer target TTR distance']
# features = ['GENETIC', 'SYNTACTIC', 'FEATURAL', 'PHONOLOGICAL', 'INVENTORY', 'GEOGRAPHIC']

data['relevance'] = 0

# assign relevances from 10 to 0
for source_lang in data['Source lang'].unique():
    source_lang_data = data[data['Source lang'] == source_lang].copy()
    source_lang_data['rank'] = source_lang_data['BLEU'].rank(method='min', ascending=False)
    top_indices = source_lang_data[source_lang_data['rank'] <= 10].index
    data.loc[top_indices, 'relevance'] = 11 - source_lang_data.loc[top_indices, 'rank']

groups = data['Source lang']
ndcg_scores = []


def func(i):
    ranker = LGBMRanker(
        boosting_type='gbdt',
        objective='lambdarank',
        n_estimators=100,
        metric='lambdarank',
        num_leaves=16,
        min_data_in_leaf=5,
        random_state=i,
        feature_fraction=0.8,
        verbose=-1
    )
    #leave one language out
    for train_idx, test_idx in logo.split(data, groups=groups):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        train_X = train_data[features]
        train_y = train_data['relevance']
        test_X = test_data[features]
        test_y = test_data['relevance']

        train_group_sizes = train_data.groupby('Transfer lang').size().tolist()

        train_dataset = lgb.Dataset(train_X, label=train_y,group=train_group_sizes)
        test_dataset = lgb.Dataset(test_X, label=test_y, group=[len(test_y)], reference=train_dataset)

        # Train
        ranker.fit(train_X, train_y, group=train_group_sizes, verbose=-1)

        # Predict and evaluate NDCG@3
        y_pred = ranker.predict(test_X)
        #linear ndcg
        ndcg = ndcg_score([test_y.values], [y_pred], k=3)
        #exp ndcg
        # ndcg = letor_metrics.ndcg_score(test_y.values,y_pred,3)
        ndcg_scores.append(ndcg)

    # Calculate the average NDCG@3 score
    print([round(float(x),3) for x in ndcg_scores])
    average_ndcg = np.mean(ndcg_scores)
    print(f'Average NDCG@3: {average_ndcg * 100}')

    #final model)
    # ranker.fit(data[features], data['relevance'], group=[53] * 54)
    # print(ranker.feature_importances_)
    # lightgbm.plot_importance(ranker, importance_type='split')
    # plt.show()
    # ranker.booster_.save_model('LightGBM_model.txt')
a = []
for i in range(10,30):
    a.append(func(i))
print(a)

