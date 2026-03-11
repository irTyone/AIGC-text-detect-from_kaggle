import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    roc_curve
)

##############################
# 参数
##############################

PRED_FILE = "outputs/m3.json"
META_FILE = "/home/liuyuan/DetectRL/filtered_eval_set.json"

OUTPUT_RANK_FILE = "outputs/rank_result.json"
OUTPUT_METRIC_FILE = "outputs/metrics.json"


##############################
# 读取数据
##############################

with open(PRED_FILE, "r", encoding="utf-8") as f:
    pred_data = json.load(f)

with open(META_FILE, "r", encoding="utf-8") as f:
    meta_data = json.load(f)

df_pred = pd.DataFrame(pred_data)
df_meta = pd.DataFrame(meta_data)

print("Prediction shape:", df_pred.shape)
print("Meta shape:", df_meta.shape)



df_meta["label_bin"] = (df_meta["label"] == "llm").astype(int)


df_pred["rank_score"] = df_pred["generated"].rank(method="average") / len(df_pred)

##############################
# 保存 rank 文件
##############################

rank_output = df_pred[["id", "generated", "rank_score"]].to_dict(orient="records")

with open(OUTPUT_RANK_FILE, "w", encoding="utf-8") as f:
    json.dump(rank_output, f, indent=2)

print("Saved rank file:", OUTPUT_RANK_FILE)


##############################
# 对齐预测与标签
##############################

# 假设 id 和 metadata 顺序一致
df_all = df_pred.copy()
df_all["label"] = df_meta["label_bin"]

y_true = df_all["label"].values
y_score = df_all["rank_score"].values


##############################
# ROC + 最优threshold
##############################

fpr, tpr, thresholds = roc_curve(y_true, y_score)

# Youden index
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

y_pred = (y_score >= optimal_threshold).astype(int)

##############################
# metrics
##############################

roc_auc = roc_auc_score(y_true, y_score)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary"
)

acc = accuracy_score(y_true, y_pred)

conf_mat = confusion_matrix(y_true, y_pred).tolist()

##############################
# 保存 metrics
##############################

result = {
    "roc_auc": float(roc_auc),
    "optimal_threshold": float(optimal_threshold),
    "conf_matrix": conf_mat,
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "accuracy": float(acc),
}

with open(OUTPUT_METRIC_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("Saved metrics:", OUTPUT_METRIC_FILE)
print(result)
