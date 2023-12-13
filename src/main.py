import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Função para avaliação dos modelos
def evaluate_model(model, X, y, folds, metric='accuracy'):
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring=metric)
    return scores


# Função para exibir métricas
def display_metrics(y_true, y_pred, average='binary'):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average=average):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average=average):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average=average):.4f}")


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

# Avaliando os modelos
folds = 5
rf_scores = evaluate_model(rf_model, X, y, folds)
gb_scores = evaluate_model(gb_model, X, y, folds)
svm_scores = evaluate_model(svm_model, X, y, folds)

# Exibindo resultados
print(f"Random Forest Scores: {rf_scores.mean():.4f}")
print(f"Gradient Boosting Scores: {gb_scores.mean():.4f}")
print(f"SVM Scores: {svm_scores.mean():.4f}")

# Treinando e avaliando Random Forest para análise mais detalhada
rf_model.fit(X, y)
rf_preds = rf_model.predict(X)

# Exibindo métricas e matriz de confusão
display_metrics(y, rf_preds, average='weighted')
sns.heatmap(confusion_matrix(y, rf_preds), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest - Confusion Matrix')
plt.show()
