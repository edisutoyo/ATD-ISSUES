# run_bert_experiments.py

import pandas as pd
from sklearn.model_selection import train_test_split
from classification.bert_classification_functions import run_experiments_bert

data = pd.read_csv("dataset.csv")  

# === Encode labels (label: ATD/Non-ATD → 1/0) ===
label_mapping = {label: idx for idx, label in enumerate(data["label"].unique())}
data["label"] = data["label"].map(label_mapping)



texts = data['Summary_Description'].tolist()
labels = data['label'].tolist()

# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Run BERT experiments
results_df = run_experiments_bert(
    xtrain=xtrain,
    ytrain=ytrain,
    xtest=xtest,
    ytest=ytest,
    min=100,                  # Start with 100 samples
    max=2300,                 # End with 1000 samples
    increments=100,           # Increase by 100 each run
    filepath="/scratch/p311371/bert_results/",  # Folder to store outputs
    numberofruns=1,           # Repeat each experiment
    modelname='bert-base-uncased',
    no_epochs=3,
    batch_size=16,
    eval_bath_size=16,
    max_length=256            # Set a max token length
)

# Save experiment results
results_df.to_csv("bert_random_experiment_results.csv", index=False)
print("Experiment results saved to 'bert_random_experiment_results.csv'")

# output_filepath = '/scratch/p311371/bert_results/'