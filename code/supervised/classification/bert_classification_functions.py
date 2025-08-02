## Functions to carry out experiments and model training with BERT ##

import pandas as pd
import random

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import numpy as np
import torch, gc
import datetime

from keras.backend import clear_session

from transformers import BertForSequenceClassification, AutoTokenizer
from transformers import AdamW, BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer


def preproccess(tokenizer, data, max_length):

    ''' Generate encodings from the tokenizer'''

    encodings = tokenizer(data, truncation=True, padding=True, max_length=max_length, add_special_tokens=True)
    return encodings

 
class dataset(torch.utils.data.Dataset):

    ''' Wrapping the tokenized data into a torch dataset'''

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def set_training_args(outputdirectory, no_epochs=10, batch_size=12, eval_batch_size=12):

    ''' Setting the training arguments for model training '''

    training_args = TrainingArguments(
    output_dir= outputdirectory,
    num_train_epochs=no_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=eval_batch_size, 
    warmup_steps=500, 
    weight_decay=0.01,
    logging_dir=outputdirectory+'/logs', 
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model = 'f1',
    evaluation_strategy="steps",
    seed = 42) 
    return training_args

def setup_trainer(model, tokenizer, training_args, train_dataset, eval_dataset):

    ''' Setting up the trainer module'''

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=eval_dataset,           # evaluation dataset
        compute_metrics=compute_metrics)     # the callback that computes metrics of interest
    return trainer

def compute_metrics(pred):

  ''' Evaluating the classifier '''

  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  target_names = ["Non-ATD", "ATD"]
  precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', pos_label=1)
  acc = accuracy_score(labels, preds)
  report = classification_report(labels, preds, target_names=target_names)
  return {'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'report': report}

def train_model(trainer):
  print('starting training')
  training = trainer.train()
  print(training)
  print('training finished')
  return trainer, training

def evaluate(trainer):
  evaluation = trainer.evaluate()
  print('starting evaluation')
  print(evaluation)
  print('Finished evaluation')
  return evaluation

def save_results(outputdirectory, trainer, evaluation, model_name, no_epochs, batch_size, training_args, training='NA'):

  ''' Saving the classification results'''  

  dateTimeObj = datetime.datetime.now()
  timestampStr = dateTimeObj.strftime("%d-%b-%Y_(%H:%M:%S.%f)")
  with open(f'{outputdirectory}/{model_name}_{timestampStr}.txt', 'w+') as f:
    f.write("Training args: \n" + "model used: " + model_name + "model_location" + model_name +
            "\n no. epochs " + str(no_epochs) + '\n batch size ' + str(batch_size) +
            '\n trainer args: \n' + str(training_args) +
            '\n training: \n' + str(training) +
            '\n Evaluation: \n' + str(evaluation))
  trainer.save_model(f'{outputdirectory}/model')

class SimpleDataset:

    ''' Generating a simple dataset from tranformer encodings'''

    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}




def train_classifier1(xtrain, # Training texts
                      ytrain, # Training labels
                      xtest, # Testing texts
                      ytest, # Testing labels
                      output_filepath,
                      epochs=10,
                      batch_size=16,
                      max_length = 400,
                      model_name= 'bert-base-uncased'):

  ''' Function to train transfomer model '''
                      
  base_model = BertForSequenceClassification.from_pretrained(model_name, classifier_dropout=0.3, num_labels = 2)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  base_model.resize_token_embeddings(len(tokenizer)) 
  training_args = set_training_args(output_filepath , epochs, batch_size)
  xtrain_embeddings = preproccess(tokenizer, xtrain, max_length)
  train_dataset = dataset(xtrain_embeddings, ytrain)
  xtest_embeddings = preproccess(tokenizer, xtest, max_length)
  test_dataset = dataset(xtest_embeddings, ytest)
  trainer = setup_trainer(base_model, tokenizer, training_args, train_dataset, test_dataset)
  trainer1, training = train_model(trainer)
  evaluation = evaluate(trainer1)
  save_results(output_filepath, trainer, evaluation, model_name, epochs, batch_size, training_args, training='NA')
  return trainer1, evaluation


def error_analysis(test_text, test_label, tokenizer, trainer):

  ''' Carrying out error analysis with define testing data and a trained model'''

  data = SimpleDataset(tokenizer(test_text, return_tensors="pt", padding=True, truncation=True))
  preds = trainer.predict(data)
  results = np.argmax(preds.predictions, axis=-1).tolist()
  target_names = ["Non-ATD", "ATD"]
  precision, recall, f1, _ = precision_recall_fscore_support(test_label, results, average='binary', pos_label=1)
  acc = accuracy_score(test_label, results)
  dataframe = pd.DataFrame()
  count = 0
  for i, result in enumerate(results):
    if result != test_label[i]:
      dataframe.loc[count, 'text'] = test_text[i]
      dataframe.loc[count, 'label'] = test_label[i]
      dataframe.loc[count, 'prediction'] = result
      count+=1
  return dataframe, {'accuracy': acc,
                     'f1': f1,
                     'precision': precision,
                     'recall': recall}



def run_experiments_bert(xtrain, # Training texts
                         ytrain, # Training labels
                         xtest,  # Testing texts
                         ytest,  # Testing labels
                         min,    # Minimum training size
                         max,    # Maximum training size
                         increments, # Number of trainining examples to increase each iteration
                         filepath, # Where results will be saved
                         numberofruns, # Number of experiment repeats
                         modelname,
                         no_epochs,
                         batch_size,
                         eval_bath_size,
                         max_length=512):

    ''' Function to carry out training size experiments and save results'''
    samples = [i for i in range(min, max+increments, increments)]
    df = pd.DataFrame()
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    xtest_embeddings = preproccess(tokenizer, xtest, max_length)
    test_dataset = dataset(xtest_embeddings, ytest)
    training_args = set_training_args(filepath,no_epochs, batch_size)
    count = 0
    for run in range(numberofruns):
        for i, sample in enumerate(samples):
             
            base_model = BertForSequenceClassification.from_pretrained(modelname,
                                                                       classifier_dropout=0.3,
                                                                       num_labels = 2)
            base_model.to('cuda')
            count +=1
            random.seed(count)
            indices = random.sample(list(range(len(xtrain))), sample)
            xnew = [xtrain[index]for index in indices]
            ynew = [ytrain[index] for index in indices]
            xnew_embeddings = preproccess(tokenizer, xnew, max_length)
            train_dataset = dataset(xnew_embeddings, ynew)
            trainer = setup_trainer(base_model, tokenizer, training_args, train_dataset, test_dataset)
            trainer1, training = train_model(trainer)
            evaluation = evaluate(trainer1)
            save_results(filepath, trainer, evaluation, modelname, no_epochs, batch_size, training_args, training='NA')
            df.loc[count, 'model'] = "bert-base-uncased"
            df.loc[count, 'run'] = run
            df.loc[count, 'training_size'] = sample
            df.loc[count, 'recall'] = evaluation['eval_recall']
            df.loc[count, 'fscore'] = evaluation['eval_f1']
            df.loc[count, 'precision'] = evaluation['eval_precision']
            df.loc[count, 'accuracy'] = evaluation['eval_accuracy']
            print("Trained model with {} training points {}/{}".format(sample, run+1, numberofruns))
            clear_session()
            torch.cuda.empty_cache()
    return df