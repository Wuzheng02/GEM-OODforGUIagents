import numpy as np
import json
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# ==== 文件路径 ====
file_path = '/data3/wuzh/OOD/AITZ_ID/train/AITZ_train_atlas_inputscore.json'
nag_paths = [
    #'/data3/wuzh/OOD/AITZ_ID/OOD_dataset/androidcontrol_inputscore.json',
    #'/data3/wuzh/OOD/AITZ_ID/OOD_dataset/Kairos_test_inputscore.json',
    #'/data3/wuzh/OOD/AITZ_ID/OOD_dataset/metagui_inputscore.json',
    #'/data3/wuzh/OOD/AITZ_ID/OOD_dataset/screenspot-mobile_inputscore.json',
    #'/data3/wuzh/OOD/AITZ_ID/OOD_dataset/omniact-desktop_inputscore.json',
    '/data3/wuzh/OOD/platforms/desktop_atlas_inputscore.json',
]
pos_path = '/data3/wuzh/OOD/AITZ_ID/test/AITZ_atlas_inputscore.json'

with open(file_path, 'r') as file:
    data = json.load(file)
input_scores = [item['input_score'] for item in data]
input_scores = np.array(input_scores).reshape(-1, 1)

bic_scores = []
possible_components = range(1, 20)
for n_components in possible_components:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(input_scores)
    bic_scores.append(gmm.bic(input_scores))

best_n_components_bic = possible_components[np.argmin(bic_scores)]


optimal_gmm = GaussianMixture(n_components=best_n_components_bic)
optimal_gmm.fit(input_scores)

means = optimal_gmm.means_.flatten()
covariances = optimal_gmm.covariances_.flatten()


boundaries = []
for i in range(optimal_gmm.n_components):
    lower_bound = means[i] - 2 * np.sqrt(covariances[i])
    upper_bound = means[i] + 2 * np.sqrt(covariances[i])
    boundaries.append((lower_bound, upper_bound))

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]: 
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return merged

merged_boundaries = merge_intervals(boundaries)


for i, (lb, ub) in enumerate(merged_boundaries):
    print(f'{i + 1}: [{lb}, {ub}]')

nag_scores = []
for path in nag_paths:
    with open(path, 'r') as file:
        nag_data = json.load(file)
        nag_scores.extend([item['input_score'] for item in nag_data])


with open(pos_path, 'r') as file:
    pos_data = json.load(file)
pos_scores = [item['input_score'] for item in pos_data]


def classify_scores(scores, merged_boundaries):
    predictions = []
    for score in scores:
        classified = any(lower <= score <= upper for (lower, upper) in merged_boundaries)
        predictions.append(classified)
    return predictions

nag_labels = classify_scores(nag_scores, merged_boundaries)
pos_labels = classify_scores(pos_scores, merged_boundaries)


y_true = [0] * len(nag_scores) + [1] * len(pos_scores)
y_pred = nag_labels + pos_labels

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print(f'{accuracy:.4f}')
print(f'{precision:.4f}')
print(f'{recall:.4f}')
print(f'{f1:.4f}')
print(conf_matrix)
