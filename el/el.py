import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRanker
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score
import letor_metrics

# Load the data
data = pd.read_csv('el.csv')
groups = data['Target lang']
logo = LeaveOneGroupOut()
# Define feature columns and target column
features = ['Entity overlap','Transfer lang dataset size','Target lang dataset size',
            'Transfer over target size ratio','GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']
# features = [
#      'Transfer lang dataset size',
#     'Target lang dataset size', 'Transfer over target size ratio','Entity overlap'
# ]
# features = ['GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']
target = 'Accuracy'

data['relevance'] = 0
for source_lang in data['Target lang'].unique():
    source_lang_data = data[data['Target lang'] == source_lang].copy()
    source_lang_data['rank'] = source_lang_data[target].rank(method='min', ascending=False)
    top_indices = source_lang_data[source_lang_data['rank'] <= 10].index
    data.loc[top_indices, 'relevance'] = 11 - source_lang_data.loc[top_indices, 'rank']

groups = data['Target lang']
ndcg_scores = []

# Parameters for LightGBM
ranker = LGBMRanker(
    boosting_type='gbdt',
    objective ='lambdarank',
    n_estimators=100,
    num_leaves=16,
    min_data_in_leaf=5,
    output_model='LightGBM_model.txt',
    random_state=50,
    verbose=0,
    feature_fraction=0.8
)



all_rankings = pd.DataFrame()

# Leave-one-source-language-out cross-validation
for train_idx, test_idx in logo.split(data, groups=groups):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    train_X = train_data[features]
    train_y = train_data['relevance']
    test_X = test_data[features]
    test_y = test_data['relevance']

    train_group_sizes = train_data.groupby('Target lang').size().tolist()
    # Prepare LightGBM dataset with query information
    train_dataset = lgb.Dataset(train_X, label=train_y,group=train_group_sizes)
    test_dataset = lgb.Dataset(test_X, label=test_y, group=[len(test_y)], reference=train_dataset)


    # Train the model
    ranker.fit(train_X, train_y, group=train_group_sizes,verbose=-1)

    # Predict and evaluate NDCG@3
    y_pred = ranker.predict(test_X)
    ndcg = ndcg_score([test_y], [y_pred], k=3)
    # ndcg = letor_metrics.ndcg_score(test_y.values, y_pred,k=3)
    ndcg_scores.append(ndcg)

group = data.groupby('Target lang').size().tolist()
ranker.fit(data[features], data['relevance'],group=group)
lgb.plot_importance(ranker, importance_type='split')
print(ranker.feature_importances_)
# plt.show()
ranker.booster_.save_model('LightGBM_model_el.txt')
# Calculate the average NDCG@3 score
print([round(float(x),4) for x in ndcg_scores])
average_ndcg = np.mean(ndcg_scores)
print(f'Average NDCG@3: {round(average_ndcg*100,1)}')