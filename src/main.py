import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


url = "https://archive.ics.uci.edu/static/public/15/data.csv"
data = pd.read_csv(url)

data = data.dropna()
data = data.drop('Sample_code_number', axis=1)

X = data.drop('Class', axis=1)
y = data['Class']

scaler = StandardScaler()
X = scaler.fit_transform(X)


def evaluate_model(model, X, y, folds=5):
    k_fold = KFold(n_splits=folds, shuffle=True, random_state=42)
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    for metric in metrics:
        score = cross_val_score(model, X, y, cv=k_fold, scoring=metric)
        metric_avg = np.mean(score)
        metric_std = np.std(score)
        print(f"\t-{metric.capitalize()} (avg): {metric_avg:.4f} ± {metric_std:.4f}")

    y_pred = cross_val_predict(model, X, y, cv=k_fold)
    cm = confusion_matrix(y, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {folds} folds')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


k_values = [2, 5, 10, 15]

print("# =============================== KNN ====================================", end='')

param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}
knn_model = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn_model, param_grid_knn, cv=5, scoring='accuracy')

grid_search_knn.fit(X, y)
best_params_knn = grid_search_knn.best_params_

knn_model = KNeighborsClassifier(**best_params_knn)

for k in k_values:
    print(f"\nKNN Scores with folds = {k}:")
    evaluate_model(knn_model, X, y, folds=k)


print("\n\n")
print("# ========================== DECISION TREE ===============================", end='')

param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_model = DecisionTreeClassifier()
grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, scoring='accuracy')

grid_search_dt.fit(X, y)
best_params_dt = grid_search_dt.best_params_

dt_model = DecisionTreeClassifier(**best_params_dt)

for k in k_values:
    print(f"\nDecision Tree Scores with folds = {k}:")
    evaluate_model(dt_model, X, y, folds=k)


print("\n\n")
print("# ========================== RANDOM FOREST ===============================", end='')

param_dist_rf = {
    'n_estimators': [50, 100, 150],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_model = RandomForestClassifier()
random_search_rf = RandomizedSearchCV(rf_model, param_distributions=param_dist_rf, n_iter=10, cv=5, scoring='accuracy', random_state=42)

random_search_rf.fit(X, y)
best_params_rf = random_search_rf.best_params_

rf_model = RandomForestClassifier(**best_params_rf)

for k in k_values:
    print(f"\nRandom Forest Scores with folds = {k}:")
    evaluate_model(rf_model, X, y, folds=k)
