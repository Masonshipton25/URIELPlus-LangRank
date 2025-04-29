# Uses Wilcoxon signed-rank test with NDCG@3 scores for URIEL and URIEL+ to determine statistical significance of results

from scipy.stats import wilcoxon

# NDCG@3 scores for URIEL experiment.
URIEL_scores = []

# NDCG@3 scores for URIEL+ experiment.
URIELPlus_scores = []

# Perform Wilcoxon signed-rank test
stat, p = wilcoxon(URIEL_scores, URIELPlus_scores)

print("Statistic:", stat)
print("p-value:", p)