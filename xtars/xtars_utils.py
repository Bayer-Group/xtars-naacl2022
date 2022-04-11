# coding=utf-8
__author__ = "Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__email__ = "angelo.ziletti@bayer.com"
__date__ = "21/02/22"

import logging
import os.path
import random

import numpy as np
import pandas as pd

from flair.data import Corpus, Sentence
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TARSClassifier

logger = logging.getLogger(__name__)


class MixinXTARS(object):
    """
    Mixin class with utils function for XTARS training and prediction
    """

    def __init__(self):
        pass

    @staticmethod
    def extract_candidates(df, cols_cand_labels):

        if isinstance(cols_cand_labels, str):
            logger.warning(
                "List of candidate label columns should be a list. Found single element"
                "Converting to list of one element."
            )
            cols_cand_labels = [cols_cand_labels]

        # extract candidate labels for XTARS sampling
        can_labels_list = []

        for idx in range(len(df)):
            can_labels = set()
            for cols in cols_cand_labels:
                cand_label = df[cols][idx]
                can_labels.add(cand_label)

            can_labels_list.append(can_labels)

        return can_labels_list

    @staticmethod
    def get_score(label):
        return label.score

    def predict_from_model(
        self, tars, df, col_text, candidate_labels, model_name="xtars"
    ):

        logger.info("Predicting zero-shot with the loaded XTARS model")

        tars_output = []
        tars_label_list_1 = []
        tars_score_list_1 = []

        for idx, input_sentence in enumerate(df[col_text].tolist()):
            candidates = list(candidate_labels[idx])
            sentence = Sentence(input_sentence)

            # predict zero-shot and pass only the candidate labels to the model
            tars.predict_zero_shot(sentence, candidates)
            tars_labels = sentence.labels

            tars_output.append(str(tars_labels))

            if len(tars_labels) > 0:
                tars_labels.sort(key=self.get_score, reverse=True)

                tars_label_list_1.append(tars_labels[0].value)
                tars_score_list_1.append(tars_labels[0].score)

            else:
                # no predicted labels returned, return nan
                tars_label_list_1.append(np.nan)
                tars_score_list_1.append(np.nan)

        df["pred_label_text_{}_1".format(model_name)] = tars_label_list_1
        df["prob_{}_1".format(model_name)] = tars_score_list_1

        df["output_{}".format(model_name)] = tars_output

        return df

    @staticmethod
    def load_tars_model(model_filename, model_folder, merge_path, task_name=None):
        model_path_tars = merge_path(model_folder, model_filename)
        logging.info("Loading model from: {}".format(model_path_tars))
        tars = TARSClassifier.load(model_path_tars)
        logging.info("List of existing tasks: {}".format(tars.list_existing_tasks()))

        if task_name is not None:
            tars.switch_to_task(task_name)
            logging.info("Switching to task {}".format(task_name))

        return tars

    def predict_tars(
        self,
        df,
        model_filenames,
        model_folder,
        col_text,
        merge_path,
        candidate_labels,
        task_name,
        model_name_prefix="xtars",
    ):

        for idx, model_filename in enumerate(model_filenames):
            logger.info(
                "Prediction for model {} ({}/{})".format(
                    model_filename, idx + 1, len(model_filenames)
                )
            )
            tars = self.load_tars_model(
                model_filename=model_filename,
                model_folder=model_folder,
                merge_path=merge_path,
                task_name=task_name,
            )

            df = self.predict_from_model(
                tars=tars,
                df=df,
                col_text=col_text,
                candidate_labels=candidate_labels,
                model_name="{}_{}".format(model_name_prefix, idx),
            )

            del tars

        return df

    @staticmethod
    def avg_predictions(
        row,
        prefix_class="pred_label_text",
        prefix_class_prob="prob",
        model_name="xtars",
        nb_models=3,
        top_ks=1,
    ):
        """Average predictions from multiple models"""

        pred_class = {}
        for idx_model in range(nb_models):
            for top_k in range(top_ks):
                col_name_class = "{}_{}_{}_{}".format(
                    prefix_class, model_name, str(idx_model), str(top_k + 1)
                )
                col_name_prob = "{}_{}_{}_{}".format(
                    prefix_class_prob, model_name, str(idx_model), str(top_k + 1)
                )

                if row.loc[col_name_class] not in pred_class:
                    pred_class[row.loc[col_name_class]] = [row.loc[col_name_prob]]
                else:
                    pred_class[row.loc[col_name_class]].append(row.loc[col_name_prob])

        pred_classes_mean = {}
        for key, value in pred_class.items():
            pred_classes_mean[key] = np.nansum(np.array(value)) / nb_models

        class_label = max(pred_classes_mean, key=pred_classes_mean.get)
        class_prob = pred_classes_mean[class_label]

        out = [class_label, class_prob, str(pred_classes_mean)]

        return pd.Series(out)

    @staticmethod
    def get_flair_embedding(string, embedding):
        sentence = Sentence(string)
        embedding.embed(sentence)
        return sentence.embedding.cpu().detach().numpy()

    def unsupervised_match_classes(self, df, label_encoder, col_text="text", top_k=5):
        classes = list(label_encoder.classes_)
        bert_embedding = TransformerDocumentEmbeddings("dmis-lab/biobert-v1.1")

        class_embed = []
        for item in classes:
            item_emb = self.get_flair_embedding(item, bert_embedding)
            class_embed.append(item_emb)

        class_embed = np.array(class_embed)

        text_embedding_series = df[col_text].apply(
            self.get_flair_embedding, args=(bert_embedding,)
        )

        text_embedding = []
        for item in text_embedding_series:
            text_embedding.append(item)

        text_embedding = np.array(text_embedding)

        sim_matrix = np.dot(text_embedding, class_embed.transpose())

        # idx_top_matches = np.argmax(sim_matrix, axis=1)
        idx_top_matches = np.argsort(-sim_matrix, axis=1)[:, :top_k]

        for top_i in range(top_k):
            matched_classes = []
            for match in idx_top_matches[:, top_i]:
                class_text = classes[match]
                matched_classes.append(class_text)

            df["pred_label_text_{}_unsupervised".format(top_i)] = matched_classes

        return df

    @staticmethod
    def get_samples(nb_samples, class_labels, templates=None):
        chosen_labels = []
        text_samples = []
        cand_label_top1 = []
        cand_label_top2 = []
        cand_label_top3 = []

        if templates is None:
            templates = [
                "This text is about",
                "I think this text is about",
                "The sentence is related to",
                "The topic of this snippet is",
            ]

        for idx in range(nb_samples):
            # define the true label randomly
            true_label = random.choice(class_labels)
            # now generate negative samples
            # remove chosen label from labels, and make a copy of possible negative samples
            cand_labels = class_labels.copy()
            cand_labels.remove(true_label)

            # sample without repetition
            neg_labels = random.sample(cand_labels, 2)

            # return the true label as top candidate label
            cand_label_top1.append(true_label)
            # return the rest of the candidate labels randomly
            cand_label_top2.append(neg_labels[0])
            cand_label_top3.append(neg_labels[1])

            text_sample = random.choice(templates) + " " + true_label

            text_samples.append(text_sample)
            chosen_labels.append(true_label)

        return (
            text_samples,
            chosen_labels,
            cand_label_top1,
            cand_label_top2,
            cand_label_top3,
        )

    def create_toy_dataset(self, output_file, max_nb_classes=50, nb_samples=100):

        class_labels = ["class_{}".format(idx) for idx in range(0, max_nb_classes)]

        (
            text_data,
            labels,
            cand_label_top1,
            cand_label_top2,
            cand_label_top3,
        ) = self.get_samples(nb_samples=nb_samples, class_labels=class_labels)

        data = zip(text_data, labels, cand_label_top1, cand_label_top2, cand_label_top3)

        df = pd.DataFrame(
            data=data,
            columns=[
                "text",
                "label",
                "cand_label_top1",
                "cand_label_top2",
                "cand_label_top3",
            ],
        )

        df.to_csv(output_file, index=False, encoding="utf8")

        return output_file

    def load_sample_data(
        self,
        data_folder,
        label_col="label",
        label_type="class",
        num_negative_labels_to_sample=3,
    ):

        # data in data folder
        train_csv = os.path.join(data_folder, "train.csv")
        val_csv = os.path.join(data_folder, "dev.csv")
        test_csv = os.path.join(data_folder, "test.csv")

        print("Loading the dataset...")

        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
        df_test = pd.read_csv(test_csv)

        print(
            "Total training classes: {}".format(
                len(df_train[label_col].unique().tolist())
            )
        )

        # define columns mapping to create FLAIR corpus
        column_name_map = {0: "text", 1: "label"}

        # init a corpus using column format, data folder and the names of the train, dev and test files
        corpus: Corpus = CSVClassificationCorpus(
            data_folder,
            column_name_map,
            skip_header=True,
            delimiter=",",
            label_type=label_type,
        )

        # define the number of negative samples
        # change the number of samples depending on the data point
        df_train["Negative_samples"] = np.random.randint(
            low=max(0, num_negative_labels_to_sample - 1),
            high=num_negative_labels_to_sample + 2,
            size=len(df_train),
        )

        # columns that contain the negative classes
        cols_cand_labels = ["cand_label_top1", "cand_label_top2", "cand_label_top3"]

        # extract candidates for each datapoint
        candidate_labels_list = self.extract_candidates(
            df=df_train, cols_cand_labels=cols_cand_labels
        )

        # go through each sentence and add the negative labels
        train_sents = []
        for sentence in corpus.train:
            train_sents.append(sentence)

        for idx, sentence in enumerate(train_sents):
            setattr(sentence, "candidate_labels", candidate_labels_list[idx])
            setattr(
                sentence,
                "nb_neg_labels_to_sample_for_this_point",
                df_train["Negative_samples"].tolist()[idx],
            )

        # make a corpus with train and test split
        # (you need to add a test set so that the code avoids to add it automatically
        corpus = Corpus(train=train_sents, dev=corpus.dev, test=corpus.test)

        label_dictionary = corpus.make_label_dictionary(label_type=label_type)

        return corpus, label_dictionary

    @staticmethod
    def get_demo_folder():
        # create a directory to save the model
        if not os.path.exists(r"./"):
            os.makedirs(r"./models_tmp")
        model_folder = r"./models_tmp"
        base_path_demo = os.path.join(model_folder, "xtars_demo")
        return base_path_demo
