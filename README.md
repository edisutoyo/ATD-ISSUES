# Replication package of Reducing Labeling Effort in Architecture Technical Debt Detection through Active Learning and Explainable AI

## Description of this study:
Self-Admitted Technical Debt (SATD) refers to technical compromises explicitly admitted by developers in natural language artifacts, such as code comments, commit messages, and issue trackers. Among its types, Architecture Technical Debt (ATD) is particularly difficult to detect due to its abstract and context-dependent nature. Manual annotation of ATD is costly, time-consuming, and challenging to scale. To reduce labeling effort, this study combines keyword-based filtering, active learning, and explainable AI for ATD detection. We refined an existing dataset of ATD{-}related Jira issues to obtain an expert-validated seed set used to extract representative keywords. These keywords were then applied to identify more than 103k candidate issues across 10 open-source projects. To assess the reliability of keyword-based filtering, we qualitatively evaluated a statistically representative sample of labeled issues. Building on the resulting dataset, we applied active learning with multiple query strategies to prioritize informative samples for annotation. The results show that Breaking Ties achieved the best performance, with an F1-score of 0.72 and a 49\% reduction in annotation effort. To improve transparency, we used SHAP and LIME to explain ATD classification results. Expert evaluation showed that both methods provided useful explanations, with LIME generally preferred for its clarity and ease of use.

## Structure of the Replication Package


```text
└── ATD-ISSUES-main
    ├── LICENSE
    ├── README.md
    ├── code
    │   ├── data prep
    │   │   ├── cs-keybert-keywords-extraction.py
    │   │   ├── keybert-keywords-extraction.py
    │   │   └── tf-idf-keywords-extraction.py
    │   ├── supervised
    │   │   ├── classification
    │   │   │   ├── active_learning.py
    │   │   │   └── bert_classification_functions.py
    │   │   ├── train_active_learning.py
    │   │   └── train_bert_random.py
    │   └── unsupervised
    │       ├── CS-BERT.py
    │       ├── KEYBERT-BERT.py
    │       └── TF-IDF.py
    ├── dataset
    │   ├── LATEST-ATD-DATASET-NO-WEAK.csv
    │   └── LATEST-ATD-DATASET.csv
    ├── evaluation_study
    │   └── Evaluation Study - Evaluating the Interpretability and Usefulness of LIME and SHAP Explanations for ATD Classification from Jira Issues - Google Forms.pdf
    └── result
        ├── atd_tfidf_keywords.csv
        ├── atd_keybert_keywords.csv
        ├── atd_cs_keybert_keywords.csv
        ├── Extracted Keywords.pdf
        └── query strategies
            ├── 14-AL-results-PredictionEntropy.csv
            ├── 16-AL-results-BreakingTies.csv
            ├── 22-AL-results-RandomSampling.csv
            ├── 22-EK-AL-results-EmbeddingKMeans.csv
            ├── 24-AL-results-LeastConfidence.csv
            └── 25-AL-results-ContrastiveActiveLearning.csv
```


## Contents

### Dataset
- `LATEST-ATD-DATASET.csv`\
    A CSV dataset derived from Jira issue trackers of ten Apache open-source projects contains issue reports labelled ATD, Weak-ATD, and Non-ATD.
- `LATEST-ATD-DATASET-NO-WEAK.csv`\
    This is a CSV dataset derived from Jira issue trackers of ten Apache open-source projects. It contains issue reports labeled as ATD and Non-ATD, with the exception of Weak-ATD.

### Code
#### `data prep/` 
  This folder contains all source code used in the **data preparation phase**

#### `unsupervised/`
  This folder includes code for detecting ATD using **three keyword-based methods**

#### `supervised/`
  This folder contains code for both **supervised learning** and **active learning** approaches to ATD detection, including model training and active learning query strategies

### Result
  This folder contains extracted keywords from three keyword-based methods and performance results in terms of precision, recall, and F1-score from four different query strategies

### Evaluation study
  This folder contains a survey form for the expert evaluation study


## How to Reproduce
### Run keyword-based (unsupervised)
To run ATD detection using three keyword-based methods:
```
python "code/unsupervised/<script_name>.py"
```

### Run supervised learning (baseline)
To train and evaluate a supervised model without active learning:
```
python "code/supervised/train_bert_random.py"
```

### Run active learning experiments
To reproduce the active learning results (including Breaking Ties strategy):
```
python "code/supervised/train_active_learning.py"
```
## Paper

Latest version available on [arXiv](https://arxiv.org/abs/2603.02944)

If you use this dataset to support your research and publish a paper, we encourage you to cite the following paper in your publication:

```
@article{sutoyo2026reducing,
  title={Reducing Labeling Effort in Architecture Technical Debt Detection through Active Learning and Explainable AI},
  author={Sutoyo, Edi and Avgeriou, Paris and Capiluppi, Andrea},
  journal={arXiv preprint arXiv:2603.02944},
  year={2026}
}
```


## Contact

- Please use the following email addresses if you have questions:
    - :email: <e.sutoyo@rug.nl>
