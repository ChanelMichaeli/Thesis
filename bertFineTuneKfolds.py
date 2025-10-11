
import torch
import random
import numpy as np
import re
import emoji
import os
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset

from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.nn import Linear, ReLU, Dropout, CrossEntropyLoss
import torch.nn.functional as F

class EnhancedBERTModel(torch.nn.Module):
    def __init__(self, original_model,class_counts):
        super(EnhancedBERTModel, self).__init__()
        self.bert = original_model.bert  # Pre-trained BERT
        self.class_counts = class_counts 
        # self.dropout1 = Dropout(0.3)
        self.dense1 = Linear(original_model.config.hidden_size, 256)
        self.relu1 = ReLU()
        # self.dropout2 = Dropout(0.3)
        self.dense2 = Linear(256, 128)
        self.relu2 = ReLU()
        self.classifier = Linear(128, 2)  # Output layer
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Pooled output [CLS] token
        # x = self.dropout1(pooled_output)
        x = self.dense1(pooled_output)
        x = self.relu1(x)
        # x = self.dropout2(x)
        x = self.dense2(x)
        x = self.relu2(x)
        logits = self.classifier(x)
        
        
        # If labels are provided, calculate the loss
        if labels is not None:
                    
            if self.class_counts is not None:
                class_weights = 1.0 / self.class_counts.float()
                class_weights = class_weights / class_weights.sum() * len(self.class_counts)  # normalize
                
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                # self.model.to(self.device)        
                    
            else:
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
                    mode="training",topic=0,print_res=True):
        
        self.model_name = model_name
        self.print_res=print_res
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
        # in the 'else' case , the model is already pre-trained and all the datasets is for tests 
        for i in range (len(lst)):
            lst[i]=absolute_path+lst[i]
        self.test_datasets=lst
        
        self.f1_binary_scores = dict()
        for i in range (len(self.test_datasets)):
            self.f1_binary_scores[self.test_datasets[i]] = []
            
        self.f1_macro_scores = dict()
        for i in range (len(self.test_datasets)):
            self.f1_macro_scores[self.test_datasets[i]] = []
                       
        self.recall_scores = dict()
        for i in range (len(self.test_datasets)):
            self.recall_scores[self.test_datasets[i]] = []  
            
        self.precision_scores = dict()
        for i in range (len(self.test_datasets)):
            self.precision_scores[self.test_datasets[i]] = []          
        
        self.accuracy = dict()
        for i in range (len(self.test_datasets)):
            self.accuracy[self.test_datasets[i]] = []   
        
        # self.set_kFolds()
        
    def set_kFolds(self):
        if(self.mode=="training"):

            train = pd.read_csv(self.dataset1)
            n=5
            kf = KFold(n_splits=n, random_state=42, shuffle=True)
            
            for train_index, _ in kf.split(train):
                train_df = train.iloc[train_index]
                inputs,labels,masks = self.load_data(train_df)
                train_dataset = CustomDataset(inputs=inputs, labels=labels, masks=masks)
                labels_tensor = torch.tensor(train_dataset.labels, dtype=torch.long)
                imbalanced= False ## we balanced the datasets.
                if imbalanced:
                    class_counts = torch.bincount(labels_tensor)
                    self.set_model(self.model_name,class_counts.to(self.device))
                else:
                    self.set_model(self.model_name,None)

                trainer = self.train(train_dataset)
                
                for i in range (len(self.test_datasets)):
                    test = pd.read_csv(self.test_datasets[i])
                    
                    inputs,labels,masks = self.load_data(test)
                    test_dataset = CustomDataset(inputs=inputs, labels=labels, masks=masks)
                    metrics = trainer.evaluate(test_dataset)
                    print(self.test_datasets[i]," metrics ",metrics)
                    # f1, recall, precision = self.evaluate_model(test,self.test_datasets[i])
                    self.f1_binary_scores[self.test_datasets[i]].append(metrics["eval_f1_binary"])
                    self.f1_macro_scores[self.test_datasets[i]].append(metrics["eval_f1_macro"])
                    self.recall_scores[self.test_datasets[i]].append(metrics["eval_recall"])
                    self.precision_scores[self.test_datasets[i]].append(metrics["eval_precision"])
                    self.accuracy[self.test_datasets[i]].append(metrics["eval_accuracy"])
            
            #save results
            res = {"test data":[],"mean f1 macro":[],"mean f1 binary":[],"mean recall":[],"mean precision":[],"mean accuracy":[],"std f1 macro":[],"std f1 binary":[],"std recall":[],"std precision":[],"std accuracy":[]}
            for i in range (len(self.test_datasets)):
                res["test data"].append(self.test_datasets[i].split('/')[-2])
                res["mean f1 macro"].append(sum(self.f1_macro_scores[self.test_datasets[i]])/n)
                res["mean f1 binary"].append(sum(self.f1_binary_scores[self.test_datasets[i]])/n)
                res["mean recall"].append(sum(self.recall_scores[self.test_datasets[i]])/n)
                res["mean precision"].append(sum(self.precision_scores[self.test_datasets[i]])/n)
                res["mean accuracy"].append(sum(self.accuracy[self.test_datasets[i]])/n)
                res["std f1 macro"].append(np.std(self.f1_macro_scores[self.test_datasets[i]]))
                res["std f1 binary"].append(np.std(self.f1_binary_scores[self.test_datasets[i]]))
                res["std recall"].append(np.std(self.recall_scores[self.test_datasets[i]]))
                res["std precision"].append(np.std(self.precision_scores[self.test_datasets[i]]))
                res["std accuracy"].append(np.std(self.accuracy[self.test_datasets[i]]))


        else:
            trainer = Trainer(
                model=self.model,                         # The model to be trained
                tokenizer=self.tokenizer,                 # Tokenizer
                compute_metrics= self.compute_metrics               # Optional: define custom metric computation
            )
            res = self.eval_model(trainer)
            self.predict(trainer)
            
        return pd.DataFrame(res)
    '''
        desc= create the model and the tokenizer
    '''
    
    def set_tokenizer(self,model_name):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create a  tokenizer and ClassificationModel
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
    
    
    def set_model(self,model_name,class_counts):
        if(self.mode=="training"):
            base_model = BertForSequenceClassification.from_pretrained(
                model_name, 
                num_labels = 2, # The number of output labels for binary classification.
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False) # Whether the model returns all hidden-states.
            self.model = EnhancedBERTModel(base_model, class_counts)

        else:
            self.model = torch.load(model_name+"/model.pth")  # Load the saved model
            # Ensure model is in evaluation mode
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
                        learning_rate=1e-5,
                        num_train_epochs=5,                  # Number of training epochs
                        per_device_train_batch_size=4,      # Training batch size
                        # warmup_steps=500,                    # Number of warmup steps
                        # weight_decay=0.01,                   # Strength of weight decay
                        logging_dir='./logs',                # Directory for storing logs
                        logging_steps=10,                    # Log every 10 steps
                        evaluation_strategy="no",         # Evaluate every epoch
                        save_strategy="no",
                        load_best_model_at_end=True,         # Load the best model when training ends
                        seed=42                              # Set seed for reproducibility
                        )
        
        self.set_random_seed(training_args.seed)
        
        trainer = Trainer(
            model=self.model,                         # The model to be trained
            args=training_args,                  # Training arguments
            train_dataset=data,         # Training dataset
            tokenizer=self.tokenizer,                 # Tokenizer
            data_collator=None,                  # Optional: custom data collator
            compute_metrics= self.compute_metrics               # Optional: define custom metric computation
        )

        trainer.train()
    
        return trainer
    
    def compute_metrics(self,pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1_binary = f1_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro')

        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_binary': f1_binary,'f1_macro':f1_macro}
        # print(metrics) 
        return metrics

    def eval_model(self,trainer):

        for i in range (len(self.test_datasets)):
            test = pd.read_csv(self.test_datasets[i])
            test = test[ (test['label'] == 1 )]
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
            test = test[ (test['label'] == 0 )]
            inputs,labels,masks = self.load_data(test)
            test_dataset = CustomDataset(inputs=inputs, labels=labels, masks=masks)
            
            predictions = trainer.predict(test_dataset)
            # Extract logits from the output
            logits = predictions.predictions
            probabilities = F.softmax(torch.tensor(logits), dim=-1)
            prob_class_1 = probabilities[:, 1].numpy() # positive class
            prob_class_0 = probabilities[:, 0].numpy() 
                    
            output_resutls = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/scores_results_finetuned/aux_labels/negative_labels/" + self.test_datasets[i].split("/")[-2] + "/" + self.test_datasets[i].split("/")[-1]
            if os.path.exists(output_resutls) and os.path.getsize(output_resutls) > 0:
                df = pd.read_csv(output_resutls)
            else:
                df = pd.DataFrame()
            if "text" not in df.columns:
                df["text"] = test["text"]
            df[f"{self.model_name.split('/')[-1]}_probs"] = prob_class_0      
            df.to_csv(output_resutls, index=False)
  
    def saveModel(self):
        if(self.output_model_dir!=None):
            model_to_save =self.model
            # model_to_save.save_pretrained(self.output_model_dir)
            path_to_save = self.output_model_dir + "/model.pth"
            self.tokenizer.save_pretrained(self.output_model_dir)
            torch.save(model_to_save,path_to_save)
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
                max_length=max_length)
                
            input_ids.append(encoded_sent)

        # Pad our input tokens with value 0.
        input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long",
                                   value=self.tokenizer.pad_token_id, truncating="pre", padding="pre")
        
        # Create attention masks
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
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
