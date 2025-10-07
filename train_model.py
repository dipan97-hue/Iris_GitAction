import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# -------------------------------------------
# 1️ Load and Clean Dataset
# -------------------------------------------
dataset = pd.read_csv('iris.csv')
print(" Dataset loaded successfully:", dataset.shape)
print(dataset.head())

# Standardize column names
dataset.columns = [col.strip(' (cm)').replace(' ', '_').lower() for col in dataset.columns]

# -------------------------------------------
# 2️ Feature Engineering
# -------------------------------------------
dataset['sepal_ratio'] = dataset['sepal_length'] / dataset['sepal_width']
dataset['petal_ratio'] = dataset['petal_length'] / dataset['petal_width']

# Keep only required columns
expected_cols = [
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
    'sepal_ratio', 'petal_ratio', 'target'
]
dataset = dataset[expected_cols]

print(" Feature engineering complete. Columns now:", list(dataset.columns))

# -------------------------------------------
# 3️ Split Data
# -------------------------------------------
X = dataset.drop('target', axis=1).astype('float32')
y = dataset['target'].astype('int32')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f" Data split: Train={X_train.shape}, Test={X_test.shape}")

# -------------------------------------------
# 4️ Logistic Regression Model
# -------------------------------------------
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
logreg.fit(X_train, y_train)
print(" Logistic Regression model trained successfully.")

prediction_lr = logreg.predict(X_test)

cm = confusion_matrix(y_test, prediction_lr)
f1 = f1_score(y_test, prediction_lr, average='micro')
recall = recall_score(y_test, prediction_lr, average='micro')
precision = precision_score(y_test, prediction_lr, average='micro')

train_acc_lr = logreg.score(X_train, y_train) * 100
test_acc_lr = logreg.score(X_test, y_test) * 100

print(f"""
🔹 Logistic Regression Metrics:
  Train Accuracy: {train_acc_lr:.4f}
  Test Accuracy: {test_acc_lr:.4f}
  Precision: {precision:.4f}
  Recall: {recall:.4f}
  F1 Score: {f1:.4f}
""")

# -------------------------------------------
# 5️ Random Forest Model
# -------------------------------------------
print(' Training Random Forest...')
Rand_rg = RandomForestClassifier()
Rand_rg.fit(X_train, y_train)
print(" Random Forest trained successfully.")

prediction_rf = Rand_rg.predict(X_test)


f1_rf = f1_score(y_test, prediction_rf, average='micro')
recall_rf = recall_score(y_test, prediction_rf, average='micro')
precision_rf = precision_score(y_test, prediction_rf, average='micro')

train_acc_rf = Rand_rg.score(X_train, y_train) * 100
test_acc_rf = Rand_rg.score(X_test, y_test) * 100

print(f"""
 Random Forest Metrics:
  Train Accuracy: {train_acc_rf:.4f}
  Test Accuracy: {test_acc_rf:.4f}
  Precision: {precision_rf:.4f}
  Recall: {recall_rf:.4f}
  F1 Score: {f1_rf:.4f}
""")

# -------------------------------------------
# 6️ Confusion Matrix Plot
# -------------------------------------------
def plot_cm(cm, target_name, title='Confusion Matrix', cmap=None, Normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    missclass = 1 - accuracy
    cmap = plt.get_cmap("Blues") if cmap is None else cmap
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_name))
    plt.xticks(tick_marks, target_name, rotation=45)
    plt.yticks(tick_marks, target_name)

    if Normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if Normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel(f'Predicted label\naccuracy={accuracy:.4f}; misclass={missclass:.4f}')
    plt.show()
    plt.close()

print(' Plotting confusion matrix for Random Forest...')
target_name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
plot_cm(cm, target_name, title='Confusion Matrix')


# -------------------------------------------
# 7️ Feature Importance Plot (fixed)
# -------------------------------------------
importances = Rand_rg.feature_importances_
labels = X.columns  # Only features, exclude target

feature_df = pd.DataFrame({'Feature': labels, 'Importance': importances})
features = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features)
plt.title('Feature Importance - Random Forest', fontsize=16)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
plt.close()


#-------------------------------------------
# 8 Saving the scores in txt
#-------------------------------------------
with open('scores.txt', 'w') as f:
    f.write(f'Random Forest - Train Accuracy: {train_acc_rf:.2f}%\n')
    f.write(f'Random Forest - Test Accuracy: {test_acc_rf:.2f}%\n')
    f.write(f'Recall: {recall_rf:.2f}%\n')
    f.write(f'F1 Score: {f1_rf:.2f}%\n')
    f.write(f'Precision: {precision_rf:.2f}%\n')
    
    f.write('\n\n')  # spacing between models
    
    f.write(f'Logistic Regression - Train Accuracy: {train_acc_lr:.2f}%\n')
    f.write(f'Logistic Regression - Test Accuracy: {test_acc_lr:.2f}%\n')
    f.write(f'Recall: {recall:.2f}%\n')
    f.write(f'F1 Score: {f1:.2f}%\n')
    f.write(f'Precision: {precision:.2f}%\n')

print(" Scores written successfully to 'scores.txt'")
