from flair.data import Sentence
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer
from xtars.xtars_model import XTARSClassifier


def test_train_tars(get_tmp_demo_folder, load_sample_data_for_test):

    corpus, label_dictionary = load_sample_data_for_test

    # create a TARS classifier
    tars = TARSClassifier(embeddings="sshleifer/tiny-distilroberta-base")

    tars.add_and_switch_to_new_task(
        task_name="question 2_CLASS",
        label_dictionary=label_dictionary,
        label_type="class",
    )

    # initialize the text classifier trainer
    trainer = ModelTrainer(tars, corpus)

    # start the training
    trainer.train(
        base_path=get_tmp_demo_folder,
        learning_rate=0.02,
        mini_batch_size=1,
        max_epochs=1,
    )

    sentence = Sentence("This is great!")
    tars.predict(sentence)


def test_train_xtars(get_tmp_demo_folder, load_sample_data_for_test):

    corpus, label_dictionary = load_sample_data_for_test

    # create a TARS classifier
    tars = XTARSClassifier(embeddings="sshleifer/tiny-distilroberta-base")

    tars.add_and_switch_to_new_task(
        task_name="question 2_CLASS",
        label_dictionary=label_dictionary,
        label_type="class",
    )

    # initialize the text classifier trainer
    trainer = ModelTrainer(tars, corpus)

    # start the training
    trainer.train(
        base_path=get_tmp_demo_folder,
        learning_rate=0.02,
        mini_batch_size=1,
        max_epochs=1,
    )

    sentence = Sentence("This is great!")
    tars.predict(sentence)