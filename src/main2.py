import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================================================
# ============================= DATASET ==================================
# ========================================================================
# Carregando os dados
url = "https://archive.ics.uci.edu/static/public/45/data.csv"
data = pd.read_csv(url)

# Visualizando as primeiras linhas dos dados
#print(data.head())

# Informações sobre os dados
#print('\nInfos:')
#print(data.info())

# Verificando a distribuição das classes
#print('\nDistribuicao das classes:')
#print(data['num'].value_counts())


# ========================================================================
# ========================== PRE PROCESSING ==============================
# ========================================================================
# Tratamento de valores faltantes (se necessário)
data = data.dropna()

# Separando features e target
X = data.drop('num', axis=1)
y = data['num']

# Normalização das features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# ========================================================================
# ============================= ANALYSIS =================================
# ========================================================================
# Função para avaliação dos modelos
def evaluate_model(model, X, y, avg='weighted', folds=5):
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    all_y_true = []
    all_y_pred = []

    y_np = y.to_numpy()

    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        # Treina e faz a predicao
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Accumulate true and predicted labels
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # Calcular métricas
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['precision'].append(precision_score(y_test, y_pred, average=avg, zero_division=1))  # Setting zero_division=1
        scores['recall'].append(recall_score(y_test, y_pred, average=avg, zero_division=1))  # Setting zero_division=1
        scores['f1'].append(f1_score(y_test, y_pred, average=avg, zero_division=1))  # Setting zero_division=1

    # Compute confusion matrix for the entire dataset
    cm = confusion_matrix(all_y_true, all_y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Exibir métricas médias e desvio padrão
    for metric in ('accuracy', 'precision', 'recall', 'f1'):
        metric_avg = np.mean(scores[metric])
        metric_std = np.std(scores[metric])
        print(f"{metric.capitalize()} (avg): {metric_avg:.4f} ± {metric_std:.4f}")

    return scores


# Função para exibir métricas
# def display_metrics(y_true, y_pred, average='binary'):
#     print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
#     print(f"Precision: {precision_score(y_true, y_pred, average=average):.4f}")
#     print(f"Recall: {recall_score(y_true, y_pred, average=average):.4f}")
#     print(f"F1 Score: {f1_score(y_true, y_pred, average=average):.4f}")


# ========================================================================
# =============================== KNN ====================================
# ========================================================================
# Definindo os hiperparâmetros a serem testados para o KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}
knn_model = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn_model, param_grid_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(X, y)
# Melhores hiperparâmetros encontrados para o KNN
best_params_knn = grid_search_knn.best_params_

knn_model = KNeighborsClassifier(**best_params_knn)

print("# ========================================================================")
print("# =============================== KNN ====================================")
print("# ========================================================================")

for folds in range(2,3):
    print(f"\nKNN Scores with folds = {folds}:")
    knn_scores = evaluate_model(knn_model, X, y, folds=folds)

    # Exibindo resultados
    print(f"Scores: {knn_scores}")


# ========================================================================
# ========================== DECISION TREE ===============================
# ========================================================================
# Definindo os hiperparâmetros a serem testados para a Decision Tree
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
# Melhores hiperparâmetros encontrados para a Decision Tree
best_params_dt = grid_search_dt.best_params_

dt_model = DecisionTreeClassifier(**best_params_dt)

print("\n\n# ========================================================================")
print("# ========================== DECISION TREE ===============================")
print("# ========================================================================")

for folds in range(2, 3):
    print(f"\nDecision Tree Scores with folds = {folds}:")
    dt_scores = evaluate_model(dt_model, X, y, folds=folds)

    # Exibindo resultados
    print(f"Scores: {dt_scores}")


# ========================================================================
# ========================== RANDOM FOREST ===============================
# ========================================================================
# Definindo os hiperparâmetros a serem testados para o Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_model = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X, y)
# Melhores hiperparâmetros encontrados para o Random Forest
best_params_rf = grid_search_rf.best_params_

rf_model = RandomForestClassifier(**best_params_rf)

print("\n\n# ========================================================================")
print("# ========================== RANDOM FOREST ===============================")
print("# ========================================================================")

for folds in range(2, 3):
    print(f"\nRandom Forest Scores with folds = {folds}:")
    rf_scores = evaluate_model(rf_model, X, y, folds=folds)

    # Exibindo resultados
    print(f"Scores: {rf_scores}")


# ========================================================================
# ========================== GRADIENT BOOSTING ===========================
# ========================================================================
# Definindo os hiperparâmetros a serem testados para o Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 1.0]
}
gb_model = GradientBoostingClassifier()
grid_search_gb = GridSearchCV(gb_model, param_grid_gb, cv=5, scoring='accuracy')
grid_search_gb.fit(X, y)
# Melhores hiperparâmetros encontrados para o Gradient Boosting
best_params_gb = grid_search_gb.best_params_

gb_model = GradientBoostingClassifier(**best_params_gb)

print("\n\n# ========================================================================")
print("# ========================= GRADIENT BOOSTING ============================")
print("# ========================================================================")

for folds in range(2, 3):
    print(f"\nGradient Boosting Scores with folds = {folds}:")
    gb_scores = evaluate_model(gb_model, X, y, folds=folds)

    # Exibindo resultados
    print(f"Scores: {gb_scores}")


# ========================================================================
# ================================= SVM ==================================
# ========================================================================
# Definindo os hiperparâmetros a serem testados para o SVM
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}
svm_model = SVC()
grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X, y)
# Melhores hiperparâmetros encontrados para o SVM
best_params_svm = grid_search_svm.best_params_

svm_model = SVC(**best_params_svm)

print("\n\n# ========================================================================")
print("# ================================ SVM ===================================")
print("# ========================================================================")

for folds in range(2, 3):
    print(f"\nSVM Scores with folds = {folds}:")
    svm_scores = evaluate_model(svm_model, X, y, folds=folds)

    # Exibindo resultados
    print(f"Scores: {svm_scores}")
