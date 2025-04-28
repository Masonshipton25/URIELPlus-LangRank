import pandas as pd
import numpy as np
from lightgbm import LGBMRanker
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score

# Load the data
data = pd.read_csv('csv_datasets\\mt.csv') 
groups = data['Source lang']
logo = LeaveOneGroupOut()

# Define feature columns and target column
# LANGRANK with language vectors and additional dataset-dependent features such as size and type-token ratio
features = ['Overlap word-level', 'Overlap subword-level', 'Transfer lang dataset size',
    'Target lang dataset size', 'Transfer over target size ratio', 'Transfer lang TTR',
    'Target lang TTR', 'Transfer target TTR distance', 'GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']

# LANGRANK with only language vectors
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

# Parameters for LightGBM
ranker = LGBMRanker(
    boosting_type='gbdt',
    objective='lambdarank',
    n_estimators=100,
    metric='lambdarank',
    num_leaves=16,
    min_data_in_leaf=5,
    random_state=50,
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

    # Train the model
    ranker.fit(train_X, train_y, group=train_group_sizes)

    # Predict and evaluate NDCG@3
    y_pred = ranker.predict(test_X)
    ndcg = ndcg_score([test_y], [y_pred], k=3)
    ndcg_scores.append(ndcg)

# Calculate the average NDCG@3 score
average_ndcg = np.mean(ndcg_scores)
print(f'Average NDCG@3: {round(average_ndcg*100,1)}')