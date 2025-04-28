import pandas as pd
import numpy as np
from lightgbm import LGBMRanker
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score

# Load the data
data = pd.read_csv('csv_datasets\\dep.csv')
logo = LeaveOneGroupOut()

# Define feature columns and target column
# LANGRANK with language vectors and additional dataset-dependent features such as size and type-token ratio
features = ['Word overlap','Transfer lang dataset size',
            'Target lang dataset size', 'Transfer over target size ratio', 'Transfer lang TTR',
            'Target lang TTR', 'Transfer target TTR distance','GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']

# LANGRANK with only language vectors
# features = [ 'GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']

data['relevance'] = 0

def ndcg1(rel_true, rel_pred):

    int_rel = rel_true.astype(int)
    rel_index = np.argsort(rel_pred)[::-1]
    true_indexed = int_rel[rel_index]
    int_rel[::-1].sort()
    dcg = (2**true_indexed[0] - 1)/np.log2(2) + (2**true_indexed[1] - 1)/np.log2(3) + (2**true_indexed[2] - 1)/np.log2(4)
    idcg = (2**int_rel[0] - 1)/np.log2(2) + (2**int_rel[1] - 1)/np.log2(3) + (2**int_rel[2] - 1)/np.log2(4)
    return dcg / idcg

# assign relevances from 10 to 0
for source_lang in data['Target lang'].unique():
    source_lang_data = data[data['Target lang'] == source_lang].copy()
    source_lang_data['rank'] = source_lang_data['Accuracy'].rank(method='dense', ascending=False)
    top_indices = source_lang_data[source_lang_data['rank'] <= 10].index
    data.loc[top_indices, 'relevance'] = 11 - source_lang_data.loc[top_indices, 'rank']

groups = data['Target lang']
ndcg_scores = []

# Parameters for ranker
ranker = LGBMRanker(
    boosting_type='gbdt',
    objective ='lambdarank',
    n_estimators=100,
    metric='lambdarank',
    num_leaves=16,
    min_data_in_leaf=5,
    verbose=-1
)

query = [29] * 29

#leave one language out
for train_idx, test_idx in logo.split(data, groups=groups):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    train_X = train_data[features]
    train_y = train_data['relevance']
    test_X = test_data[features]
    test_y = test_data['relevance']

    # Train the model
    ranker.fit(train_X, train_y, group=query, eval_set=[(test_X, test_y)], eval_group=[[29]], eval_at=[3])

    # Predict and evaluate NDCG@3
    y_pred = ranker.predict(test_X)
    score = ndcg_score([test_y],[y_pred],k=3)
    ndcg_scores.append(score)

# Calculate the average NDCG@3 score
average_ndcg = np.mean(ndcg_scores)
print(f'Average NDCG@3: {round(average_ndcg*100,1)}')