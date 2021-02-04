import torch
from sklearn import metrics
import numpy as np
import pandas as pd
from pathlib import Path

pred = pd.read_csv(Path("../submissions/effi7a_noise_student.csv"))
label = pd.read_csv(Path("../data/dataset/test/label.csv"))
y_pred = list(pred['diagnosis'])
y_true = list(label['diagnosis'])

ACC = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
print(len(y_pred))
print(len(y_true))
print(ACC)
