import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
# Load training and evaluation data
label_file = "/home/pierres/Projet_S7/recognAItion/data_final/label_equi.csv"
sample_file = "/home/pierres/Projet_S7/recognAItion/data/sample_equi.csv"

samples_eval = "/home/pierres/Projet_S7/recognAItion/data/sample_eval.csv"
label_eval = "/home/pierres/Projet_S7/recognAItion/data/label_eval.csv"

# Load datasets
train_data = pd.read_csv(label_file)
sample_data = pd.read_csv(sample_file)

eval_data = pd.read_csv(samples_eval)
eval_label_data = pd.read_csv(label_eval)

L= []
for k in range(1,30):
    M = []
    for l in range(10):
        clf = tree.DecisionTreeClassifier(max_depth = k)
        clf = clf.fit(sample_data,train_data)
        m = clf.score(eval_data,eval_label_data)
        M.append(m)
    avg = np.mean(M)
    L.append(avg)

plt.plot([k for k in range(1,30)],L)
    
plt.show()