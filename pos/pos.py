import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRanker
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score
import letor_metrics
# Load the data
data = pd.read_csv('pos.csv')
# data = pd.read_csv('pos_updated.csv')

logo = LeaveOneGroupOut()
# Define feature columns and target column

features = ['Overlap word-level','Transfer lang dataset size','Target lang dataset size','Transfer over target size ratio',
            'Transfer lang TTR','Target lang TTR','Transfer target TTR distance',
            'GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']
# features = ['Overlap word-level','Transfer lang dataset size','Target lang dataset size','Transfer over target size ratio',
#             'Transfer lang TTR','Target lang TTR','Transfer target TTR distance']
features = ['GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']
data['relevance'] = 0

# assign relevances from 10 to 0
for source_lang in data['Task lang'].unique():
    source_lang_data = data[data['Task lang'] == source_lang].copy()
    source_lang_data['rank'] = source_lang_data['Accuracy'].rank(method='min', ascending=False)
    top_indices = source_lang_data[source_lang_data['rank'] <= 10].index
    data.loc[top_indices, 'relevance'] = 11 - source_lang_data.loc[top_indices, 'rank']

groups = data['Task lang']
ndcg_scores = []
def func(s):
    # Parameters for ranker
    ranker = LGBMRanker(
        boosting_type='gbdt',
        objective='lambdarank',
        n_estimators=100,
        metric='lambdarank',
        num_leaves=16,
        min_data_in_leaf=5,
        verbose=-1,
        feature_fraction=1,
        random_state=s
    )

    query = [60] * 11 + [59] * 15
    # query = [1] * 1545
    #leave one language out
    for train_idx, test_idx in logo.split(data, groups=groups):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        train_X = train_data[features]
        train_y = train_data['relevance']
        test_X = test_data[features]
        test_y = test_data['relevance']

        train_group_sizes = train_data.groupby('Task lang').size().tolist()

        # Prepare LightGBM dataset
        train_dataset = lgb.Dataset(train_X, label=train_y,group=train_group_sizes)
        test_dataset = lgb.Dataset(test_X, label=test_y, group=[len(test_y)], reference=train_dataset)


        # Train the model
        ranker.fit(train_X, train_y, group=train_group_sizes,verbose=-1)

        # Predict and evaluate NDCG@3
        y_pred = ranker.predict(test_X)
        ndcg = ndcg_score([test_y], [y_pred], k=3)
        # ndcg = letor_metrics.ndcg_score(test_y.values,y_pred,k=3)
        ndcg_scores.append(ndcg)

    #final model
    ranker.fit(data[features], data['relevance'], group=query)
    print(ranker.feature_importances_)
    lgb.plot_importance(ranker, importance_type='split')
    # plt.show()
    print([round(float(x),3) for x in ndcg_scores])
    print(round(np.mean(ndcg_scores)*100,1))
    return round(np.mean(ndcg_scores)*100, 1)

# old = []
# for i in range(10, 30):
#     old.append(func(i))
# print(old)

print(func(1))

