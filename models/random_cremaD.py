from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
label_file = "/home/pierres/Projet_S7/recognAItion/data/labels.csv"
sample_file = "/home/pierres/Projet_S7/recognAItion/data/sample.csv"

samples_eval = "/home/pierres/Projet_S7/recognAItion/data/sample_eval.csv"
label_eval = "/home/pierres/Projet_S7/recognAItion/data/label_eval.csv"

# Load datasets
Y = pd.read_csv(label_file)
X = pd.read_csv(sample_file)

Y = np.array(Y)

Y=Y.ravel()

eval_data = pd.read_csv(samples_eval)
eval_label_data = pd.read_csv(label_eval)
L= []
for k in range(1,30):
    M = []
    for l in range(2):
      
        clf = RandomForestClassifier(max_depth=k)
        clf.fit(X,Y)
        m = clf.score(eval_data,eval_label_data)
        M.append(m)
    avg = np.mean(M)
    L.append(avg)

plt.plot([k for k in range(1,30)],L)
    
plt.show()