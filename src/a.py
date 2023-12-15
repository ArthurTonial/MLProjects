import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Valores de k que você deseja testar
k_values = [2, 5, 10, 15]

# Dados fictícios para acurácia, precisão, recall e F1-score
accuracy_values = np.random.rand(len(k_values))
precision_values = np.random.rand(len(k_values))
recall_values = np.random.rand(len(k_values))
f1_score_values = np.random.rand(len(k_values))

# Criar gráficos separados para cada métrica
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

plt.figure(figsize=(15, 10))

for i, metric in enumerate(metrics, start=1):
    # Substituir espaços por underscores ou remover espaços
    metric_name = metric.replace(" ", "_").lower()

    plt.subplot(2, 2, i)
    plt.plot(k_values, eval(f"{metric_name}_values"), marker='o', color='b', label=metric)
    plt.title(f'{metric} vs. Number of Folds (k)')
    plt.xlabel('Number of Folds (k)')
    plt.ylabel(f'{metric} Value')
    plt.legend()

plt.tight_layout()
plt.show()
