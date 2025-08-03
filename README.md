# ATD-ISSUES
Replication package of Reducing Labeling Effort in Architecture Technical Debt Detection through Active Learning and Explainable AI

## Description of this study:
Self-Admitted Technical Debt (SATD) refers to technical compromises explicitly admitted by developers in natural language artifacts such as code comments, commit messages, and issue trackers. Among its types, Architecture Technical Debt (ATD) is particularly difficult to detect due to its abstract and context-dependent nature. Manual annotation of ATD is costly, time-consuming, and challenging to scale. This study focuses on reducing labeling effort in ATD detection by combining keyword-based filtering with active learning and explainable AI. We refined an existing dataset of 116 ATD-related Jira issues from prior work, producing 57 expert-validated items used to extract representative keywords. These were applied to identify over 103,000 candidate issues across ten open-source projects. To assess the reliability of this keyword-based filtering, we conducted a qualitative evaluation of a statistically representative sample of labeled issues. Building on this filtered dataset, we applied active learning with multiple query strategies to prioritize the most informative samples for annotation. Our results show that the Breaking Ties strategy consistently improves model performance, achieving the highest F1-score of 0.72 while reducing the annotation effort by 49%. In order to enhance model transparency, we applied SHAP and LIME to explain the outcomes of automated ATD classification. Expert evaluation revealed that both LIME and SHAP provided reasonable explanations, with the usefulness of the explanations often depending on the relevance of the highlighted features. Notably, experts preferred LIME overall for its clarity and ease of use.

## Contents

### Dataset
- `LATEST-ATD-DATASET.csv`\
    A CSV dataset derived from Jira issue trackers of ten Apache open-source projects contains issue reports labelled ATD, Weak-ATD, and Non-ATD.
- `LATEST-ATD-DATASET-NO-WEAK.csv`\
    This is a CSV dataset derived from Jira issue trackers of ten Apache open-source projects. It contains issue reports labeled as ATD and Non-ATD, with the exception of Weak-ATD.

### Source code

#### `data prep/` 
  This folder contains all source code used in the **data preparation phase**

#### `unsupervised/`
  This folder includes code for detecting ATD using **three keyword-based methods**

#### `supervised/`
  This folder contains code for both **supervised learning** and **active learning** approaches to ATD detection, including model training and active learning query strategies

### Results
  This folder contains extracted keywords from three keyword-based methods and performance results in terms of precision, recall, and F1-score from four different query strategies
