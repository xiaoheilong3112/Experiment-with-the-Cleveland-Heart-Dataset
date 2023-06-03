#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import permutation_importance
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, names=column_names, na_values='?')
df = df.dropna()

# Split the data into features and target label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y = y.apply(lambda x: 1 if x > 0 else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
confusion_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print the evaluation metrics
print('Confusion Matrix:', confusion_mat)
print('Accuracy:', accuracy)
print('Recall:', recall)
print('Precision:', precision)
print('F1 Score:', f1)
print('ROC AUC Score:', roc_auc)

# In[ ]:


importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print('Feature ranking:')

for f in range(X.shape[1]):
    print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title('Feature importances')
plt.bar(range(X.shape[1]), importances[indices], color='r', yerr=std[indices], align='center')
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# Plot the partial dependence plots
plot_partial_dependence(clf, X_train, [0, 1, 2, 3, 4, 5])
plt.show()

# Generate 10 new samples and predict the outcomes
new_samples = pd.DataFrame(np.random.randint(0,100,size=(10, 13)), columns=list('ABCDEFGHIJKLM'))
new_predictions = clf.predict(new_samples)
print('New predictions:', new_predictions)

# Generate the environment dependencies
!pip freeze > requirements.txt

# In[ ]:


!pip install umap-learn

# In[ ]:


# Import necessary libraries
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
from sklearn.inspection import plot_partial_dependence
from sklearn.manifold import TSNE, LocallyLinearEmbedding
import umap

# Predict the outcomes for the test set
y_pred = clf.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='.0f', linewidths=.5, square = True, cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix', size = 15)
plt.show()

# Compute the accuracy, recall, precision and F1 score
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy:', accuracy)
print('Recall:', recall)
print('Precision:', precision)
print('F1 Score:', f1)

# Compute the ROC curve and AUC score
fpr, tpr, _ = roc_curve(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Plot the feature importances
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
plt.figure()
plt.title('Feature importances')
plt.bar(range(X.shape[1]), importances[indices], color='r', yerr=std[indices], align='center')
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# In[ ]:


# Import necessary libraries
from sklearn.manifold import TSNE, LocallyLinearEmbedding
import umap

# Compute the t-SNE embedding
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# Plot the t-SNE embedding
plt.figure()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.title('t-SNE embedding')
plt.show()

# Compute the UMAP embedding
umap_emb = umap.UMAP(n_components=2)
X_umap = umap_emb.fit_transform(X)

# Plot the UMAP embedding
plt.figure()
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y)
plt.title('UMAP embedding')
plt.show()

# Compute the LLE embedding
lle = LocallyLinearEmbedding(n_components=2)
X_lle = lle.fit_transform(X)

# Plot the LLE embedding
plt.figure()
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=y)
plt.title('LLE embedding')
plt.show()