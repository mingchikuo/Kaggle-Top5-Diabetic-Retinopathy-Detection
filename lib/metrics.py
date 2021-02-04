import torch
from sklearn import metrics
import numpy as np

# def my_metrics(y_pred, y_true):
#     if torch.is_tensor(y_pred):
#         y_pred = y_pred.data.cpu().numpy()
#     if torch.is_tensor(y_true):
#         y_true = y_true.data.cpu().numpy()
#     if y_pred.shape[1] == 1:
#         y_pred = y_pred[:, 0]
#     else:
#         y_pred = np.argmax(y_pred, axis=1)
#     # return metrics.fbeta_score(y_true=y_true, y_pred=y_pred, beta=1.67)
#     try:
#         result = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
#     except ValueError:
#         return 0.86
#
#     # print('\t', 'y_pred:', y_pred)
#     # print('\t', 'y_true:', y_true)
#     # print('\t', 'score:', result)
#     return result
#
#
#
#
#





def my_metrics(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    else:
        y_pred = np.argmax(y_pred, axis=1)

    # metrix = {
    #     'Acc':metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
    #     'Recall': metrics.recall_score(y_true=y_true, y_pred=y_pred),
    #     'F(1.25)-Score':metrics.fbeta_score(y_true=y_true, y_pred=y_pred, beta=1.25),
    #     'Cohen Kappa':metrics.cohen_kappa_score(y1=y_true, y2=y_pred)
    # }
    #
    # print('\n', metrix['Acc'], '\n', metrix['Recall'], '\n', metrix['F(1.25)-Score'], '\n', metrix['Cohen Kappa'])
    # return metrics.fbeta_score(y_true=y_true, y_pred=y_pred, beta=1.5)

    return metrics.cohen_kappa_score(y1=y_true, y2=y_pred, weights='quadratic')
    # return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
