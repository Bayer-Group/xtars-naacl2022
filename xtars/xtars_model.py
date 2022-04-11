# coding=utf-8
__author__ = "Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__email__ = "angelo.ziletti@bayer.com"
__date__ = "21/02/22"

import gc
import logging

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import minmax_scale

import torch
from flair.data import Sentence
from flair.models import TARSClassifier

log = logging.getLogger("flair")


class XTARSClassifier(TARSClassifier):
    """
    XTARSClassifier
    """

    def __init__(
        self,
        use_xtars_sampling=True,
        top_k_embed=20,
        temperature=0.01,
        max_class_embed=500,
        neg_vs_pos_ratio=None,
        **tars_args,
    ):

        super(XTARSClassifier, self).__init__(**tars_args)

        # XTARS-specific parameters
        self.use_xtars_sampling = use_xtars_sampling
        self.temperature = temperature
        self.top_k_embed = top_k_embed
        self.max_class_embed = max_class_embed
        if neg_vs_pos_ratio is None:
            self.neg_vs_pos_ratio = self.num_negative_labels_to_sample
        else:
            self.neg_vs_pos_ratio = neg_vs_pos_ratio

        if self.use_xtars_sampling:
            log.info("Using XTARS sampling")
            log.info("Temperature: {}".format(self.temperature))
            log.info("top_k_embed: {}".format(self.top_k_embed))
            log.info("max_class_embed: {}".format(self.max_class_embed))
            log.info("neg_vs_pos_ratio: {}".format(self.neg_vs_pos_ratio))

    def _compute_label_similarity_for_current_epoch(self):
        """
        Compute the similarity between all labels for better sampling of negatives
        This method overwrite the methods present in TARSClassifier
        """

        all_labels = [
            label.decode("utf-8")
            for label in self.get_current_label_dictionary().idx2item
        ]
        log.info("Number of labels in label dictionary: {}".format(len(all_labels)))

        # split the labels in chunks, and calculate their representation separately
        # to avoid a memory error due to the (possibly) large number of labels
        n_chunks = int(len(all_labels) / self.max_class_embed) + 1
        log.info("Splitting {} labels in {} chunks.".format(len(all_labels), n_chunks))
        all_labels_list = np.array_split(np.array(all_labels), n_chunks)

        # get the label representation by looping over the chunks
        encodings_np_list = []
        for idx, labels_chunk in enumerate(all_labels_list):
            log.info("Iteration: {}/{}".format(idx + 1, n_chunks))

            label_sentences = [Sentence(label) for label in labels_chunk]

            self.tars_model.document_embeddings.embed(label_sentences)

            encodings_np = [
                sentence.get_embedding().cpu().detach().numpy()
                for sentence in label_sentences
            ]

            encodings_np_list.extend(encodings_np)

            # delete the objects to save GPU memory
            del encodings_np, label_sentences
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        encodings_np = np.array(encodings_np_list)

        # normalize the label embeddings
        normalized_encoding = minmax_scale(encodings_np)

        log.info("Computing label similarity based on embedding...")

        # compute similarity matrix (use all cores for parallel computation)
        # similarity_matrix = cosine_similarity(normalized_encoding)
        similarity_matrix_emb = pairwise_kernels(
            normalized_encoding, metric="cosine", n_jobs=-1
        )

        log.info("Computing label probabilities...")

        negative_label_probabilities = {}

        # loop over labels
        for row_index, label in enumerate(all_labels):
            negative_label_probabilities[label] = {}
            sim_row_emb = similarity_matrix_emb[row_index]

            # for a given class label, we return only the most similar top-k(top_k_embed) classes
            # this speeds-up the computation for the negative label probabilities
            # note: the top-k returned are not ordered, since ordering is not needed
            top_k_indices_unsort = np.argpartition(sim_row_emb, -self.top_k_embed)[
                -self.top_k_embed :
            ]
            top_k_labels = list(set([all_labels[i] for i in top_k_indices_unsort]))

            # get the indices according to the order the labels have in all_labels
            top_k_indices = [
                i for i, e in enumerate(all_labels) if e in set(top_k_labels)
            ]
            top_k_labels_with_idx = zip(top_k_indices, top_k_labels)

            # set to zero all label similarities with labels that are not in the top-k
            for idx in range(len(all_labels)):
                if idx not in top_k_indices:
                    sim_row_emb[idx] = 0.0

            # compute the temperature-scaled softmax row-wise
            m = torch.nn.Softmax(dim=0)
            sim_row_softmax = m(torch.tensor(sim_row_emb / self.temperature)).numpy()

            # use softmax result as sampling probability
            for column_index, other_label in top_k_labels_with_idx:
                if label != other_label:
                    negative_label_probabilities[label][other_label] = sim_row_softmax[
                        column_index
                    ]

        log.info("Negative label probabilities computed.")

        self.label_nearest_map = negative_label_probabilities

    def _get_nearest_labels_for(self, labels, neg_to_samples=None):
        """
        This overwrites the method in TARSClassifier because it creates a copy of the
        positive samples based on the number of negative labels.
        :param labels:
        :param neg_to_samples:
        :return:
        """

        already_sampled_negative_labels = set()
        if neg_to_samples is None:
            neg_to_samples = self.neg_to_samples

        for label in labels:
            plausible_labels = []
            plausible_label_probabilities = []
            for plausible_label in self.label_nearest_map[label]:
                if (
                    plausible_label in already_sampled_negative_labels
                    or plausible_label in labels
                ):
                    continue
                else:
                    plausible_labels.append(plausible_label)
                    plausible_label_probabilities.append(
                        self.label_nearest_map[label][plausible_label]
                    )

            # make sure the probabilities always sum up to 1
            plausible_label_probabilities = np.array(
                plausible_label_probabilities, dtype="float64"
            )

            plausible_label_probabilities += 1e-08
            plausible_label_probabilities /= np.sum(plausible_label_probabilities)

            if len(plausible_labels) > 0:
                num_samples = min(neg_to_samples, len(plausible_labels))

                sampled_negative_labels = np.random.choice(
                    plausible_labels,
                    num_samples,
                    replace=False,
                    p=plausible_label_probabilities,
                )
                already_sampled_negative_labels.update(sampled_negative_labels)

        return already_sampled_negative_labels

    def _get_tars_formatted_sentences(self, sentences):
        """
        Overwrite the method defined in FewshotClassifier

        :param sentences:
        :return:
        """
        label_text_pairs = []

        all_labels = [
            label.decode("utf-8")
            for label in self.get_current_label_dictionary().idx2item
        ]

        for idx, sentence in enumerate(sentences):
            label_text_pairs_for_sentence = []
            candidate_labels = []

            if self.training and self.num_negative_labels_to_sample is not None:
                positive_labels = {label.value for label in sentence.get_labels()}

                # remove the true labels from the candidate labels for negative sampling
                try:
                    candidate_neg_labels = sentence.candidate_labels - positive_labels
                except AttributeError as err:
                    log.error(err)
                    raise Exception(
                        "You need to pass the candidate labels for each datapoint for XTARS"
                    )

                # if the nb of negative samples can be found in the sentence, use it, otherwise use the
                # default specified in XTARSClassifier
                try:
                    neg_to_samples = sentence.nb_neg_labels_to_sample_for_this_point
                except AttributeError:
                    neg_to_samples = self.num_negative_labels_to_sample

                sampled_negative_labels_from_prob = self._get_nearest_labels_for(
                    positive_labels, neg_to_samples=neg_to_samples
                )

                # candidate labels are always picked so we do not need the probability for them
                sampled_negative_labels = candidate_neg_labels.union(
                    sampled_negative_labels_from_prob
                )

                pos_samples_duplication_factor = max(
                    int(neg_to_samples / self.neg_vs_pos_ratio), 1
                )

                for _ in range(pos_samples_duplication_factor):
                    for label in positive_labels:
                        label_text_pairs_for_sentence.append(
                            self._get_tars_formatted_sentence(label, sentence)
                        )

                for label in sampled_negative_labels:
                    label_text_pairs_for_sentence.append(
                        self._get_tars_formatted_sentence(label, sentence)
                    )

            else:
                for label in all_labels:
                    label_text_pairs_for_sentence.append(
                        self._get_tars_formatted_sentence(label, sentence)
                    )
            label_text_pairs.extend(label_text_pairs_for_sentence)
        return label_text_pairs
