import os.path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import xtars.xtars_utils

mixin_utils = xtars.xtars_utils.MixinXTARS()

# define some parameters for training
label_col = "label"
label_type = "class"
num_negative_labels_to_sample = 3

# define folders needed for training
data_folder = r"./sample_data/"
# create a directory to save the model
if not os.path.exists(r"./"):
    os.makedirs(r"./models_tmp")
model_folder = r"./models_tmp"
base_path_tars = os.path.join(model_folder, "xtars_demo")
output_file = os.path.join(model_folder, "preds.csv")

# data in data folder
test_csv = os.path.join(data_folder, "test.csv")

print("Loading the dataset...")

df = pd.read_csv(test_csv)

num_candidates = 3
cols_cand_labels = [
    "cand_label_top{}".format(item) for item in range(1, num_candidates + 1)
]
candidate_labels = mixin_utils.extract_candidates(
    df=df, cols_cand_labels=cols_cand_labels
)

# model_filenames = ['best-model.pt', 'final-model.pt'] #multiple models can be passed
model_filenames = ["final-model.pt"]

df = mixin_utils.predict_tars(
    df=df,
    model_filenames=model_filenames,
    model_folder=base_path_tars,
    col_text="text",
    merge_path=os.path.join,
    candidate_labels=candidate_labels,
    task_name="text classification",
    model_name_prefix="xtars",
)

df[
    ["pred_label_text_xtars_avg", "pred_label_text_xtars_avg_prob", "pred_class_avg"]
] = df.apply(
    mixin_utils.avg_predictions,
    nb_models=len(model_filenames),
    top_ks=1,
    model_name="xtars",
    axis=1,
)

acc_llt = accuracy_score(df["pred_label_text_xtars_avg"].values, df["label"].values)
print("Accuracy: {0:.4f}".format(acc_llt))

df["Correct"] = np.where((df["pred_label_text_xtars_avg"] == df["label"]), True, False)

print("Sample of predictions:")
print(df.head())
df.to_csv(output_file, index=False, encoding="utf8")

print("Predictions written to file {}".format(output_file))
