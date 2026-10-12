
import torch
import random
import numpy as np
import re
import emoji
import os
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset

from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig,
)
from torch.nn import Linear, ReLU, Dropout, CrossEntropyLoss
import torch.nn.functional as F

class EnhancedBERTModel(torch.nn.Module):
    def __init__(self, original_model):
        super(EnhancedBERTModel, self).__init__()
        self.bert = original_model.base_model  # Pre-trained transformer backbone (BERT, RoBERTa, ...)
        # self.dropout1 = Dropout(0.3)
        self.dense1 = Linear(original_model.config.hidden_size, 256)
        self.relu1 = ReLU()
        # self.dropout2 = Dropout(0.3)
        self.dense2 = Linear(256, 128)
        self.relu2 = ReLU()
        self.classifier = Linear(128, 2)  # Output layer

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Architectures without a pooler head (e.g. RoBERTa) return
        # pooler_output=None; fall back to the first token's hidden state
        # ([CLS]/<s>), which is what those models' own classification heads use.
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        # x = self.dropout1(pooled_output)
        x = self.dense1(pooled_output)
        x = self.relu1(x)
        # x = self.dropout2(x)
        x = self.dense2(x)
        x = self.relu2(x)
        logits = self.classifier(x)
        
        
        # If labels are provided, calculate the loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)  # Assuming 2 classes
            return loss, logits
            
        return logits
    

class CustomDataset(Dataset):
    def __init__(self, inputs, labels, masks):
        self.inputs = inputs
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx],
            'attention_mask': self.masks[idx],
            'labels': self.labels[idx]
        }

class BERTKfold:
    
    '''
        parameters:
            mode: there are two passable modes: "training" or "testing"
            topic: does topic cloumn exist in the dataset. "0" stand for not, "1" stand for yes.
    '''
    def __init__(self,model_name,absolute_path,datasets=None,output_model_dir=None, 
                    mode="training",topic=0,print_res=True,n_folds=5):
        
        self.model_name = model_name
        self.print_res=print_res
        self.n_folds=n_folds
        self.save_predictions = False
        self.predictions_dir = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/scores_results_finetuned/aux_labels/"
        if output_model_dir is not None:
            self.set_output_model_dir(datasets,output_model_dir)
        else:
            self.output_model_dir=None
        
        self.mode=mode # "training" or "testing"

        self.tests_results=dict()
        self.classification_report=dict()
        self.predictied_labels_for_each_ds=dict()
        self.true_labels_for_each_ds=dict()
        self.topic=topic
        
        self.set_datasets(datasets,absolute_path,mode)
        # self.set_model(model_name)
        self.set_tokenizer(model_name)
        # self.dataset_load = LoadDataset(tokenizer=self.tokenizer, mode='training')

    ######################
    #                    #
    #       getter       #
    #                    #
    ######################
    
    def get_model_name(self):
        absolute_path=self.model_name # for example '/content/.../toxcityBERT'
        relative_path=absolute_path.split("/")[-1]
        return relative_path
    
    def get_mode(self):
        return self.mode
    
    
    def get_test_dataset(self):
        return self.test_datasets
    
    '''
        desc=return classficition report of each test dataset
    '''
    def get_classifiction_report(self):
        return self.classification_report
    
    '''
        desc= self.tests_results is dictionary where his keys is the datasets full name
            and his values is the sarc results where include two kinds of results
            the first is array of probilities for each sentence of the test dataset
            and the second is the mean which called "sarc score"
    '''
    def get_sarcScore_results(self):
        return self.tests_results
    
    '''
        desc= return predictied labels for each dataset in the test datasets
    '''
    def get_predicted_labels(self):
        return self.predictied_labels_for_each_ds
    
    
    def get_true_labels_for_each_ds(self):
        return self.true_labels_for_each_ds
    ######################
    #                    #
    #       getter       #
    #        end         #
    ######################
    
    def set_output_model_dir(self,datasets_names,output_model_dir):
        lst=datasets_names.split(" ")
        first_dataset=lst[0] #for example "toxicity_dataset_2.csv"
        dataset_name=first_dataset.split("/")[0] # [toxicity,dataset,2] 
 
        #we set the folder name to be associated to training dataset
        self.output_model_dir=output_model_dir+ "/" + dataset_name    
        
    '''
        parameter = string of datasets paths divided by ";"
        desc= split the string of datasets
              where the first dataset is for training and the other is for tests
    '''
    
    def set_datasets(self,datasets,absolute_path,mode):
        lst=datasets.split(" ")
        if(mode=="training"):
            self.dataset1=absolute_path+lst[0]
            lst=lst[1:] # remove the train datasets
            # key for the in-domain held-out CV fold, kept distinct from any
            # cross-domain test_datasets path so it gets its own row/scores
            self.heldout_key = self.dataset1 + " ::cv_heldout"
        # in the 'else' case , the model is already pre-trained and all the datasets is for tests
        for i in range (len(lst)):
            lst[i]=absolute_path+lst[i]
        self.test_datasets=lst

        # all evaluation targets tracked per fold: the in-domain held-out
        # fold (true CV) plus any cross-domain test_datasets
        score_keys = list(self.test_datasets)
        if mode == "training":
            score_keys = [self.heldout_key] + score_keys

        self.f1_binary_scores = dict()
        for key in score_keys:
            self.f1_binary_scores[key] = []

        self.f1_macro_scores = dict()
        for key in score_keys:
            self.f1_macro_scores[key] = []

        self.recall_scores = dict()
        for key in score_keys:
            self.recall_scores[key] = []

        self.precision_scores = dict()
        for key in score_keys:
            self.precision_scores[key] = []

        self.accuracy = dict()
        for key in score_keys:
            self.accuracy[key] = []

        # self.set_kFolds()
        
    def set_kFolds(self):
        if(self.mode=="training"):

            train_full = pd.read_csv(self.dataset1)
            n=self.n_folds
            skf = StratifiedKFold(n_splits=n, random_state=42, shuffle=True)

            # cross-domain test sets never change across folds and are never
            # trained on, so tokenize them once up front
            cross_domain_datasets = dict()
            for path in self.test_datasets:
                test = pd.read_csv(path)
                inputs, labels, masks = self.load_data(test)
                cross_domain_datasets[path] = CustomDataset(inputs=inputs, labels=labels, masks=masks)

            eval_keys = [self.heldout_key] + list(self.test_datasets)

            for train_index, heldout_index in skf.split(train_full, train_full['label']):
                train_df = train_full.iloc[train_index]
                heldout_df = train_full.iloc[heldout_index]

                inputs,labels,masks = self.load_data(train_df)
                train_dataset = CustomDataset(inputs=inputs, labels=labels, masks=masks)
                self.set_model(self.model_name)

                trainer = self.train(train_dataset)

                # true in-domain CV metric: the fold held out of training
                inputs,labels,masks = self.load_data(heldout_df)
                heldout_dataset = CustomDataset(inputs=inputs, labels=labels, masks=masks)
                metrics = trainer.evaluate(heldout_dataset)
                print(self.heldout_key," metrics ",metrics)
                self.f1_binary_scores[self.heldout_key].append(metrics["eval_f1_binary"])
                self.f1_macro_scores[self.heldout_key].append(metrics["eval_f1_macro"])
                self.recall_scores[self.heldout_key].append(metrics["eval_recall"])
                self.precision_scores[self.heldout_key].append(metrics["eval_precision"])
                self.accuracy[self.heldout_key].append(metrics["eval_accuracy"])

                # cross-domain metric: same fixed external dataset(s) every
                # fold, never part of train_index above
                for path in self.test_datasets:
                    metrics = trainer.evaluate(cross_domain_datasets[path])
                    print(path," metrics ",metrics)
                    self.f1_binary_scores[path].append(metrics["eval_f1_binary"])
                    self.f1_macro_scores[path].append(metrics["eval_f1_macro"])
                    self.recall_scores[path].append(metrics["eval_recall"])
                    self.precision_scores[path].append(metrics["eval_precision"])
                    self.accuracy[path].append(metrics["eval_accuracy"])

            #save results
            res = {"test data":[],"mean f1 macro":[],"mean f1 binary":[],"mean recall":[],"mean precision":[],"mean accuracy":[],"std f1 macro":[],"std f1 binary":[],"std recall":[],"std precision":[],"std accuracy":[]}
            for key in eval_keys:
                display_name = (self.dataset1.split('/')[-2] + " (CV heldout)") if key == self.heldout_key else key.split('/')[-2]
                res["test data"].append(display_name)
                res["mean f1 macro"].append(sum(self.f1_macro_scores[key])/n)
                res["mean f1 binary"].append(sum(self.f1_binary_scores[key])/n)
                res["mean recall"].append(sum(self.recall_scores[key])/n)
                res["mean precision"].append(sum(self.precision_scores[key])/n)
                res["mean accuracy"].append(sum(self.accuracy[key])/n)
                res["std f1 macro"].append(np.std(self.f1_macro_scores[key]))
                res["std f1 binary"].append(np.std(self.f1_binary_scores[key]))
                res["std recall"].append(np.std(self.recall_scores[key]))
                res["std precision"].append(np.std(self.precision_scores[key]))
                res["std accuracy"].append(np.std(self.accuracy[key]))


        elif self.mode == "testing":
            self.set_model(self.model_name)
            trainer = Trainer(
                model=self.model,
                processing_class=self.tokenizer,
                compute_metrics=self.compute_metrics
            )
            res = self.eval_model(trainer)
            if self.save_predictions:
                self.predict(trainer)

        elif self.mode == "affective_eval":
            # Load saved model, evaluate only positive-class samples, return
            # per-sample predicted-class probabilities (saved to predictions_dir).
            self.set_model(self.model_name)
            trainer = Trainer(
                model=self.model,
                processing_class=self.tokenizer,
            )
            self.affective_eval_model(trainer)
            return pd.DataFrame()   # per-sample CSVs are written inside affective_eval_model

        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Expected 'training', 'testing', or 'affective_eval'.")

        return pd.DataFrame(res)
    '''
        desc= create the model and the tokenizer
    '''
    
    def set_tokenizer(self,model_name):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create a tokenizer and ClassificationModel. AutoTokenizer resolves
        # the right tokenizer class for whichever architecture model_name is
        # (BERT, RoBERTa, ...); for "testing"/"affective_eval" mode model_name
        # is a local dir saved by older BERT-only runs, so keep BertTokenizer there.
        if self.mode == "training":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)


    def set_model(self,model_name):
        if(self.mode=="training"):
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels = 2, # The number of output labels for binary classification.
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False) # Whether the model returns all hidden-states.
            self.model = EnhancedBERTModel(base_model)

        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            
            # Load saved state_dict
            state_dict_path = os.path.join(model_name, "model_state_dict.pth")
            if os.path.exists(state_dict_path):
                # The checkpoint was saved from EnhancedBERTModel, so we must
                # build that same architecture before loading the state dict.
                config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
                base_model = BertForSequenceClassification(config)
                enhanced_model = EnhancedBERTModel(base_model)

                state_dict = torch.load(state_dict_path, map_location="cpu")
                enhanced_model.load_state_dict(state_dict)
                self.model = enhanced_model
                
            elif os.path.exists(os.path.join(model_name, "model.pth")):
                self.model = torch.load(os.path.join(model_name, "model.pth"), map_location="cpu")
                
            else:
                raise FileNotFoundError(
                    f"No model found in {model_name}. "
                    "Expected model_state_dict.pth or model.pth"
                )
                
            # Set model to evaluation mode
            self.model.eval()
        
    def set_random_seed(self,seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setup


    def train(self,data):
        # Training arguments
        training_args = TrainingArguments(
                        output_dir=self.output_model_dir,
                        learning_rate=2e-5,
                        num_train_epochs=4,                  # Number of training epochs
                        per_device_train_batch_size=4,      # Keep small to fit 4GB GPU
                        gradient_accumulation_steps=8,      # Effective batch size = 4×8 = 32
                        # warmup_steps=500,                    # Number of warmup steps
                        # weight_decay=0.01,                   # Strength of weight decay
                        logging_dir='./logs',                # Directory for storing logs
                        logging_steps=10,                    # Log every 10 steps
                        eval_strategy="no",
                        save_strategy="no",
                        load_best_model_at_end=False,        # Requires save_strategy != "no"; disabled to avoid ValueError
                        seed=42                              # Set seed for reproducibility
                        )
        
        self.set_random_seed(training_args.seed)
        
        trainer = Trainer(
            model=self.model,                         # The model to be trained
            args=training_args,                  # Training arguments
            train_dataset=data,         # Training dataset
            processing_class=self.tokenizer,          # Tokenizer
            data_collator=None,                  # Optional: custom data collator
            compute_metrics= self.compute_metrics               # Optional: define custom metric computation
        )

        trainer.train()
    
        return trainer
    
    def compute_metrics(self,pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1_binary = f1_score(labels, preds, zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)

        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_binary': f1_binary,'f1_macro':f1_macro}
        # print(metrics) 
        return metrics

    def eval_model(self,trainer):

        for i in range (len(self.test_datasets)):
            test = pd.read_csv(self.test_datasets[i])
            inputs,labels,masks = self.load_data(test)
            test_dataset = CustomDataset(inputs=inputs, labels=labels, masks=masks)
            
            metrics = trainer.evaluate(test_dataset)
            print(self.test_datasets[i]," metrics ",metrics)
            # f1, recall, precision = self.evaluate_model(test,self.test_datasets[i])
            self.f1_macro_scores[self.test_datasets[i]] = metrics["eval_f1_macro"]
            self.f1_binary_scores[self.test_datasets[i]] = metrics["eval_f1_binary"]
            self.recall_scores[self.test_datasets[i]] = metrics["eval_recall"]
            self.precision_scores[self.test_datasets[i]] = metrics["eval_precision"]
            self.accuracy[self.test_datasets[i]] = metrics["eval_accuracy"]
            
        #save results
        res = {"test data":[],"f1_macro":[],"f1_binary":[],"recall":[],"precision":[],"accuracy":[]}
        for i in range (len(self.test_datasets)):
            res["test data"].append(self.test_datasets[i].split('/')[-2])
            res["f1_macro"].append(self.f1_macro_scores[self.test_datasets[i]])
            res["f1_binary"].append(self.f1_binary_scores[self.test_datasets[i]])
            res["recall"].append(self.recall_scores[self.test_datasets[i]])
            res["precision"].append(self.precision_scores[self.test_datasets[i]])
            res["accuracy"].append(self.accuracy[self.test_datasets[i]])
        
        return res

    def predict(self,trainer):
        for i in range (len(self.test_datasets)):
            test = pd.read_csv(self.test_datasets[i])
            inputs,labels,masks = self.load_data(test)
            test_dataset = CustomDataset(inputs=inputs, labels=labels, masks=masks)
            
            predictions = trainer.predict(test_dataset)
            # Extract logits from the output
            logits = predictions.predictions
            probabilities = F.softmax(torch.tensor(logits), dim=-1)
            prob_class_1 = probabilities[:, 1].numpy() # positive class
            prob_class_0 = probabilities[:, 0].numpy() 
                    
            output_results = os.path.join(self.predictions_dir, self.test_datasets[i].split("/")[-2], self.test_datasets[i].split("/")[-1])
            os.makedirs(os.path.dirname(output_results), exist_ok=True)

            if os.path.exists(output_results) and os.path.getsize(output_results) > 0:
                df = pd.read_csv(output_results)
            else:
                df = pd.DataFrame()
            if "text" not in df.columns:
                df["text"] = test["text"]
            df[f"{self.model_name.split('/')[-1]}_probs"] = prob_class_1      
            df.to_csv(output_results, index=False)
  
    def affective_eval_model(self, trainer):
        '''
        For each test dataset:
          - Keep only positive-class (label==1) samples.
          - Run inference with the saved model.
          - Record, for every sample, the predicted class and the
            probability the model assigned to the positive class (class 1),
            regardless of which class was predicted.
        Results are merged into self.predictions_dir / <dataset_folder> / <dataset_file>.csv,
        adding/overwriting only this model's own columns and preserving any others
        already written there by previous runs (mirrors predict()'s merge behavior).
        '''
        model_short = self.model_name.split('/')[-1]

        for i in range(len(self.test_datasets)):
            test = pd.read_csv(self.test_datasets[i])

            positive_df = test[test['label'] == 1].reset_index(drop=True)
            if len(positive_df) == 0:
                print(f"[affective_eval] No positive samples in {self.test_datasets[i]}, skipping.")
                continue

            inputs, labels, masks = self.load_data(positive_df)
            dataset = CustomDataset(inputs=inputs, labels=labels, masks=masks)

            predictions = trainer.predict(dataset)
            logits = predictions.predictions                              # shape (N, 2)
            probabilities = F.softmax(torch.tensor(logits), dim=-1).numpy()

            prob_predicted = probabilities[:, 1]                         # probability of class 1

            output_path = os.path.join(
                self.predictions_dir,
                self.test_datasets[i].split("/")[-2],
                self.test_datasets[i].split("/")[-1]
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                out_df = pd.read_csv(output_path)
                if len(out_df) != len(positive_df):
                    raise ValueError(
                        f"Existing predictions file {output_path} has {len(out_df)} rows but "
                        f"the current positive-sample subset has {len(positive_df)} rows; "
                        "refusing to merge mismatched rows."
                    )
            else:
                out_df = pd.DataFrame()
            if "text" not in out_df.columns:
                out_df["text"] = positive_df['text'].values
            if "true_label" not in out_df.columns:
                out_df["true_label"] = positive_df['label'].values
            out_df[f"{model_short}_probs"] = prob_predicted

            out_df.to_csv(output_path, index=False)
            print(f"[affective_eval] {len(positive_df)} positive samples → {output_path}")

    def saveModel(self):
        if(self.output_model_dir!=None):
            # model_to_save.save_pretrained(self.output_model_dir)
            path_to_save = self.output_model_dir + "/model_state_dict.pth"
            torch.save(self.model.state_dict(),path_to_save)
            self.tokenizer.save_pretrained(self.output_model_dir)
            print(" model saved in ",path_to_save,sep="\t")
      
    def preprocess_text(self, text):
        text = re.sub(r'#([^ ]*)', r'\1', text)
        text = re.sub(r'https.*[^ ]', 'URL', text)
        text = re.sub(r'http.*[^ ]', 'URL', text)
        text = re.sub(r'@([^ ]*)', '@USER', text)
        text = emoji.demojize(text)
        text = re.sub(r'(:.*?:)', r' \1 ', text)
        text = re.sub(' +', ' ', text)
        return text

    # Tokenizing the sentences and creating attention masks
    def load_data_and_tokenized(self, sents, labels):
        input_ids = []
        attention_masks = []
        max_length = 100
        # Preprocessing
        sents = [self.preprocess_text(str(sent)) for sent in sents]
        
        for sent in sents:
            encoded_sent = self.tokenizer.encode(
                sent,               
                add_special_tokens=True,  
                max_length=max_length,
                truncation=True)
                
            input_ids.append(encoded_sent)

        # Pad at the end so [CLS] stays at position 0 (BERT's pooler_output reads
        # position 0). Truncate at the end too, so [CLS] is never dropped.
        input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long",
                                   value=self.tokenizer.pad_token_id, truncating="post", padding="post")
        
        # Create attention masks (1 for real tokens, 0 for padding)
        for sent in input_ids:
            att_mask = [int(token_id != self.tokenizer.pad_token_id) for token_id in sent]
            attention_masks.append(att_mask)

        return input_ids, labels, attention_masks


    # Wrap the data into DataLoader, which is required by Trainer
    # def wrap_dataset(self, inputs, labels, masks, shuffle=True):
    #     # Convert to tensors
    #     inputs = torch.tensor(inputs)
    #     labels = torch.tensor(labels)
    #     masks = torch.tensor(masks)
    #     return {"input_ids": inputs, "labels": labels, 'attention_mask': masks}
        # data = TensorDataset(inputs, masks, labels)
        # # sampler = RandomSampler(data) if shuffle else SequentialSampler(data)
        # return data
        # return DataLoader(data, batch_size=8, sampler=sampler)

    # Load data and prepare DataLoader
    def load_data(self, dataset):
        sents = dataset['text'].tolist()
        labels = dataset['label'].tolist()

        # Tokenize and wrap data into DataLoader
        inputs, labels, attention_masks = self.load_data_and_tokenized(sents, labels)
        return inputs, labels, attention_masks
        # Wrap the dataset into DataLoader
        # if self.mode == "training":
        #     self.train_dataloader = self.wrap_dataset(inputs, labels, attention_masks, shuffle=True)
        # else:  # testing mode
        #     self.test_dataloader = self.wrap_dataset(inputs, labels, attention_masks, shuffle=False)
