import os.path

import flair
import xtars.xtars_utils
from flair.trainers import ModelTrainer
from torch.optim.adam import Adam
from xtars.xtars_model import XTARSClassifier

flair.set_seed(0)

mixin_utils = xtars.xtars_utils.MixinXTARS()

# define folders needed for training
data_folder = r"./sample_data/"

model_folder = mixin_utils.get_demo_folder()

# define parameters for training
num_negative_labels_to_sample = 3
label_type = "class"

corpus, label_dictionary = mixin_utils.load_sample_data(data_folder, label_col="label", label_type=label_type,
                                                        num_negative_labels_to_sample=num_negative_labels_to_sample)

xtars_clf = XTARSClassifier(
    multi_label=False,
    embeddings="allenai/scibert_scivocab_uncased",
    use_xtars_sampling=True,
    top_k_embed=3 * num_negative_labels_to_sample,
    temperature=0.1,
    neg_vs_pos_ratio=5,
    label_type=label_type,
    label_dictionary=label_dictionary,
    num_negative_labels_to_sample=num_negative_labels_to_sample,
)

print("Starting XTARS {} calculation".format(xtars_clf))
print("List of existing tasks: {}".format(xtars_clf.list_existing_tasks()))
print("Training...")

# add the task
xtars_clf.add_and_switch_to_new_task(
    task_name="text classification",
    label_type=label_type,
    label_dictionary=label_dictionary,
)

trainer = ModelTrainer(xtars_clf, corpus)

trainer.train(
    base_path=base_path_tars,  # path to store the model artifacts
    optimizer=Adam,
    learning_rate=5.0e-5,
    mini_batch_size=32,
    max_epochs=2,
    anneal_factor=0.0,
    patience=20,
    write_weights=False,
    save_final_model=True,
    shuffle=True,
    checkpoint=False,
    embeddings_storage_mode="gpu",
)

print("Model saved to {}".format(base_path_tars))
