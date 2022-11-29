# XTARS: zero/few-shot learning for large-scale text classification

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This repository contains the code of the following paper:

    @inproceedings{ziletti-etal-2022-medical,
    title = "{M}edical Coding with Biomedical Transformer Ensembles and Zero/Few-shot Learning",
    author = "Ziletti, Angelo  and
    Akbik, Alan  and
    Berns, Christoph  and
    Herold, Thomas  and
    Legler, Marion  and
    Viell, Martina",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Industry Track",
    month = jul,
    year = "2022",
    address = "Hybrid: Seattle, Washington + Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-industry.21",
    doi = "10.18653/v1/2022.naacl-industry.21",
    pages = "176--187",
    abstract = "Medical coding (MC) is an essential pre-requisite for reliable data retrieval and reporting. Given a free-text \textit{reported term} (RT) such as {``}pain of right thigh to the knee{''}, the task is to identify the matching \textit{lowest-level term} (LLT) {--}in this case {``}unilateral leg pain{''}{--} from a very large and continuously growing repository of standardized medical terms. However, automating this task is challenging due to a large number of LLT codes (as of writing over $80\,000$), limited availability of training data for long tail/emerging classes, and the general high accuracy demands of the medical domain.With this paper, we introduce the MC task, discuss its challenges, and present a novel approach called xTARS that combines traditional BERT-based classification with a recent zero/few-shot learning approach (TARS). We present extensive experiments that show that our combined approach outperforms strong baselines, especially in the few-shot regime. The approach is developed and deployed at Bayer, live since November 2021. As we believe our approach potentially promising beyond MC, and to ensure reproducibility, we release the code to the research community.",
    }

Within this paper, we present a novel approach called **XTARS** that combines traditional BERTbased classification with a recent zero/few-shot
learning approach (TARS, by [Halder et al. (2020)](https://kishaloyhalder.github.io/pdfs/tars_coling2020.pdf)).   
**XTARS** is suitable for classification tasks with very large label sets and long-tailed distribution of labels in data points.


## Installation

We recommend to create a virtual python 3.8 environment (for instance, with conda: https://docs.anaconda.com/anaconda/install/linux/), and then execute

Install latest version from the master branch on Github by:
```
git clone <GITHUB-URL>    
cd xtars    
python setup.py install     
```

## Quick start
The `XTARSClassifier` in this repository can be used in the same way as the `TARSClassifier` in [Flair](https://github.com/flairNLP/flair).

Documentation on the usage of the `TARSClassifier` in Flair can be found [here](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_10_TRAINING_ZERO_SHOT_MODEL.md).

To use XTARS instead of TARS, simply substitute `TARSClassifier` with `XTARSClassifier` at training time.    
An example of training is presented in `main_train.py`. 

During prediction, a saved `XTARSClassifier` can be loaded in exactly the same way as the `TARSClassifier`.
We refer you to the [Flair](https://github.com/flairNLP/flair) documentation for more details.

---------------

## Example code

A script for fine tuning (`main_train.py`) and making predictions (`main_predict.py`) are provided for your convenience.

### Data for Fine Tuning

Sample data are provided in the `/sample_data/` folder. 
If you are using your own data, it must be formatted as the sample data provided.    
As prescribed by [Flair](https://github.com/flairNLP/flair), to  create the corpus three files are needed (see [here](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_6_CORPUS.md)):
```
train.csv
dev.csv
test.csv
```

We prepared a *sample dataset* in `/sample_data/` for your convenience.

### Fine Tuning

Use the `main_train.py` script to fine tune a model on the sample data provided.

```
python main_train.py 
```

### Predictions

After you trained a model, you can use `main_predict.py` script to obtain prediction for the test set.

```
python main_predict.py 
```

