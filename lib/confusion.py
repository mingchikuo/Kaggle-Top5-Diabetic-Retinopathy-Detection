import pandas as pd
import argparse
from pathlib import Path
import torch
from sklearn import metrics
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_path', default='./submissions/efficientnet-b7_11021223.csv',                   help='model name: (default: arch+timestamp)')
    parser.add_argument('--label_path', default='./data/dataset/test/label.csv')
    args = parser.parse_args()
    return args

def confusion(submission_path, label_path):
    pred_AUC = pd.read_csv(Path(submission_path))
    label_AUC = pd.read_csv(Path(label_path))
    y_pred = list(pred_AUC['diagnosis'])
    y_true = list(label_AUC['diagnosis'])
    submission = pd.read_csv(Path(submission_path))
    submission = [diag for diag in submission['diagnosis']]
    label = pd.read_csv(Path(label_path))
    label = [tag for tag in label['diagnosis']]
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(label)):
        if label[i] == 1:
            if submission[i] == label[i]:
                TP += 1
            else:
                FN += 1
        else:
            if submission[i] == label[i]:
                TN += 1
            else:
                FP += 1

    print('FN:', FN, '個')
    print('FP:', FP, '個')
    print('TN:', TN, '個')
    print('TP:', TP, '個')
    ACC = 100 * ((TP+TN)/len(label))
    Precision = 100*(TP/(TP+FP))
    Sensitivity = 100*(TP/(TP+FN))
    Specificity = 100*(TN/(FP+TN))
    F = (2*(Precision * Sensitivity))/(Precision + Sensitivity)
    AUC = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
    print('Accuracy:', round(ACC, 4), '%')
    print('Precision:', round(Precision, 4), '%')
    print('Sensitivity:', round(Sensitivity, 4), '%')
    print('Specificity:', round(Specificity, 4), '%')
    print('F1-Score:', round(F, 4), '%')
    print('AUC:', round(AUC, 4))

if __name__ == '__main__':
    args = parse_args()
    confusion(args.submission_path, args.label_path)

