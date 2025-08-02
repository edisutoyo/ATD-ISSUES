import os
import logging
import pandas as pd
import transformers
import numpy as np
import pickle


from transformers import BertTokenizerFast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from keras.backend import clear_session
from transformers import AdamW

from small_text.active_learner import PoolBasedActiveLearner
from small_text.initialization import random_initialization
from small_text.query_strategies import EmbeddingKMeans, LeastConfidence
from small_text.integrations.transformers.datasets import TransformersDataset
from small_text.integrations.transformers import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory


def get_transformers_dataset(tokenizer, data, labels=None, max_length=512, unlabeled=False):
    data_out = []
    for i, doc in enumerate(data):
        encoded_dict = tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            return_attention_mask=True,
            return_tensors='pt',
            truncation='longest_first'
        )
        label = -1 if unlabeled else labels[i]
        data_out.append((encoded_dict['input_ids'], encoded_dict['attention_mask'], label))
    return TransformersDataset(data_out)


def initialize_active_learner(active_learner, dataset, true_labels, numberofsamples=50):
    x_indices_initial = random_initialization(true_labels, n_samples=numberofsamples)
    y_initial = true_labels[x_indices_initial]

    # Set all labels to -1, then apply initial labels
    new_labels = np.full(len(true_labels), fill_value=-1)
    new_labels[x_indices_initial] = y_initial
    dataset._y = new_labels  # set label directly into dataset

    active_learner.initialize_data(x_indices_initial, y_initial)
    return x_indices_initial


def evaluate(active_learner, test):
    y_pred_test = active_learner.classifier.predict(test)
    test_acc = accuracy_score(test.y, y_pred_test)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        test.y, y_pred_test, average='macro', zero_division=0
    )
    print('Test accuracy: {:.2f}'.format(test_acc))
    print('Macro Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(precision, recall, fscore))
    return (test_acc, precision, recall, fscore, None)


# def save_active_leaner(active_learner, filepath, iteration, query):
#     filepath_model = filepath + "active_learner_{}_{}.pkl".format(iteration, query)
#     active_learner.save(filepath_model)
#     print("Model disimpan: active_learner_{}_{}.pkl".format(iteration, query))

def save_active_leaner(active_learner, filepath, iteration, query):
    os.makedirs(filepath, exist_ok=True)

    # Save active learner
    filepath_model = filepath + f"active_learner_{iteration}_{query}.pkl"
    active_learner.save(filepath_model)
    print(f"Model disimpan: active_learner_{iteration}_{query}.pkl")

    # Save trained classifier (move to CPU first)
    classifier = active_learner.classifier
    classifier.model = classifier.model.to('cpu')  # move to CPU before pickling

    filepath_classifier = filepath + f"trained_classifier_{iteration}_{query}.pkl"
    with open(filepath_classifier, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Classifier disimpan: trained_classifier_{iteration}_{query}.pkl")


def fullpipeline(train_idx,
                 train_text,
                 test_idx,
                 test_text,
                 querysize,
                 numberofqueries,
                 query_strategy,
                 transformer_model,
                 filepath,
                 num_classes,
                 batch_size,
                 no_epochs):

    transformers.logging.get_verbosity = lambda: logging.NOTSET
    tokenizer = BertTokenizerFast.from_pretrained(transformer_model)

    train = get_transformers_dataset(tokenizer, train_text, unlabeled=True)
    test = get_transformers_dataset(tokenizer, test_text, labels=test_idx)

    transformer_model_args = TransformerModelArguments(transformer_model)
    clf_factory = TransformerBasedClassificationFactory(
        transformer_model_args,
        num_classes,
        kwargs={
            'device': 'cuda',
            'mini_batch_size': batch_size,
            'early_stopping_no_improvement': 5,
            'num_epochs': no_epochs
        }
    )

    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)

    model = active_learner.classifier
    if model is not None:
        optimizer = AdamW(model.model.parameters(), lr=3e-5)
    else:
        print("The model has not been initialized properly!")

    labeled_indices = initialize_active_learner(
        active_learner, train, np.array(train_idx), numberofsamples=querysize
    )

    indices = labeled_indices.tolist()
    texts = [train_text[i] for i in indices]

    active_learners = {0: active_learner}
    results = [evaluate(active_learner, test)]

    for i in range(numberofqueries):
        q_indices = active_learner.query(num_samples=querysize)

        if len(q_indices) == 0:
            print(f"[INFO] Iteration #{i + 1} did not find any data to query. Skip.")
            continue

        y = np.array(train_idx)[q_indices]
        active_learner.update(y)

        indices.extend(q_indices.tolist())
        texts.extend([train_text[idx] for idx in q_indices])

        active_learners[i + 1] = active_learner

        print('Iteration #{} ({} samples in total)'.format(i + 1, len(indices)))
        results.append(evaluate(active_learner, test))

    df = pd.DataFrame()
    df['training_size'] = [querysize * (i + 1) for i in range(len(results))]
    df['accuracy'] = [item[0] for item in results]
    df['precision'] = [item[1] for item in results]
    df['recall'] = [item[2] for item in results]
    df['fscore'] = [item[3] for item in results]
    df['model'] = [str(query_strategy) for _ in results]

    df.to_csv(filepath + 'AL-results-{}.csv'.format(str(query_strategy)), escapechar='\r')

    max_value = df['fscore'].max()
    best_performer = list(df['fscore']).index(max_value)

    save_active_leaner(active_learners[best_performer], filepath, best_performer, query_strategy)

    labelled_df = pd.DataFrame(columns=['index', 'key', 'Summary_Description_Cleaned', 'label'])
    labelled_df['index'] = indices
    labelled_df['key'] = indices
    labelled_df['text'] = texts
    labelled_df['label'] = [train_idx[i] for i in indices]
    labelled_df.to_csv(filepath + "active_learning_labels.csv", escapechar='\r')

    clear_session()
    return labelled_df, df
