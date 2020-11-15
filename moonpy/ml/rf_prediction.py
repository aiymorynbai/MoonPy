import pandas as pd
import rasterio
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (confusion_matrix,
                             accuracy_score,
                             cohen_kappa_score,
                             precision_recall_fscore_support,
                             classification_report)


def predict_extended(df, clf):
    def convert_to_uint8(arr):
        return arr.astype(np.uint8)
    
    dtype_classes = rasterio.dtypes.get_minimum_dtype(clf.classes_)
    
    probs = clf.predict_proba(df.values)
    pred_idx = probs.argmax(axis=1)
    pred = np.zeros_like(pred_idx).astype(dtype_classes)
    for i in range(probs.shape[1]):
        pred[pred_idx == i] = clf.classes_[i]
    # get reliability layers (maximum probability and margin, i.e. maximum probability minus second highest probability)
    probs_sorted = np.sort(probs, axis=1)
    max_prob = probs_sorted[:, probs_sorted.shape[1] - 1]
    margin = (
        probs_sorted[:, probs_sorted.shape[1] - 1] - probs_sorted[:, probs_sorted.shape[1] - 2]
    )

    probs = convert_to_uint8(probs * 100)
    max_prob = convert_to_uint8(max_prob * 100)
    margin = convert_to_uint8(margin * 100)

    ndigits = len(str(max(clf.classes_)))
    prob_names = [f"prob_{cid:0{ndigits}d}" for cid in clf.classes_]
    df_result = pd.concat(
        [
            pd.DataFrame({"pred": pred, "max_prob": max_prob, "margin": margin}),
            pd.DataFrame(probs, columns=prob_names),
        ],
        axis=1,
    )
    return df_result

def get_save_classification_report(true_values, predicted_values, output_csv_path):
    cl_acc = classification_report(true_values, predicted_values, output_dict=True, digits=2)
    df_acc = pd.DataFrame.from_dict(cl_acc).T
    df_acc[['f1-score','precision','recall']] = df_acc[['f1-score','precision','recall']].round(2)
    df_acc[['support']] = df_acc[['support']].astype(int)
    df_acc.to_csv(output_csv_path, sep=';')
    return df_acc