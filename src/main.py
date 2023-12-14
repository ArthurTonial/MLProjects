import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Função para avaliação dos modelos com cálculo de métricas
def evaluate_model(model, X, y, k_folds, metrics=('accuracy', 'precision', 'recall', 'f1'), average='weighted'):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    y = np.array(y)

    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcular métricas
        for metric in metrics:
            if metric == 'accuracy':
                scores['accuracy'].append(accuracy_score(y_test, y_pred))
            elif metric == 'precision':
                scores['precision'].append(precision_score(y_test, y_pred, average=average))
            elif metric == 'recall':
                scores['recall'].append(recall_score(y_test, y_pred, average=average))
            elif metric == 'f1':
                scores['f1'].append(f1_score(y_test, y_pred, average=average))

        # Exibir métricas médias e desvio padrão
        for metric in metrics:
            metric_avg = np.mean(scores[metric])
            metric_std = np.std(scores[metric])
            print(f"{metric.capitalize()} (avg): {metric_avg:.4f} ± {metric_std:.4f}")

    return scores


# Carregando os dados
url = "https://archive.ics.uci.edu/static/public/45/data.csv"  # Substitua pelo link real
data = pd.read_csv(url)

# Visualizando as primeiras linhas dos dados
print(data.head())

# Informações sobre os dados
print(data.info())

# Verificando a distribuição das classes
print(data['num'].value_counts())

# Tratamento de valores faltantes (se necessário)
data = data.dropna()

# Separando features e target
X = data.drop('num', axis=1)
y = data['num']

# Normalização das features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Inicializando os modelos
rf_model = RandomForestClassifier()
gb_model = GradientBoostingClassifier()
svm_model = SVC()

# Avaliando os modelos com cálculo de métricas
folds = 2
print('\n => Random Forest:')
rf_scores = evaluate_model(rf_model, X, y, folds, metrics=('accuracy', 'precision', 'recall', 'f1'))
print('\n => Gradient Boosting:')
gb_scores = evaluate_model(gb_model, X, y, folds, metrics=('accuracy', 'precision', 'recall', 'f1'))
print('\n => SVC:')
svm_scores = evaluate_model(svm_model, X, y, folds, metrics=('accuracy', 'precision', 'recall', 'f1'))


# # Treinando e avaliando Random Forest para análise mais detalhada
# rf_model.fit(X, y)
# rf_preds = rf_model.predict(X)
#
# # Exibindo métricas e matriz de confusão
# sns.heatmap(confusion_matrix(y, rf_preds), annot=True, fmt='d', cmap='Blues')
# plt.title('Random Forest - Confusion Matrix')
# plt.show()
