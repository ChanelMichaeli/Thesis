'''

%%capture
!pip install pytorch-pretrained-bert
!pip install transformers
!pip install datasets


'''

import torch

import random
import time
from datetime import datetime
from tqdm import tqdm, trange
import numpy as np
import re
import emoji

from transformers import  BertForSequenceClassification, BertTokenizer
from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset

from torch.utils.tensorboard import SummaryWriter

from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd

import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report

from scipy.special import softmax


class BERTModelTest:
    
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
        
        self.set_datasets(datasets,absolute_path,mode)
        self.set_model(model_name)
        self.mode=mode # "training" or "testing"
    
        self.tests_results=dict()
        self.classification_report=dict()
        self.predictied_labels_for_each_ds=dict()
        self.true_labels_for_each_ds=dict()
        self.topic=topic
        
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
        first_dataset_without_extension=first_dataset.split(".")[0] # toxicity_dataset_2
        list_of_first_dataset_words=first_dataset_without_extension.split("_") # [toxicity,dataset,2] 
        dataset_topic=list_of_first_dataset_words[0]
        dataset_num=""
        for elem in list_of_first_dataset_words:
            if (elem.isnumeric()):
                dataset_num=elem
        
        name=dataset_topic+"BERT"+dataset_num 
        #we set the folder name to be associated to training dataset
        self.output_model_dir=output_model_dir+ "/" + name  
        
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
        
    '''
        desc= create the model and the tokenizer
    '''
    def set_model(self,model_name):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create a  tokenizer and ClassificationModel
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
                model_name, 
                num_labels = 2, # The number of output labels for binary classification.
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False) # Whether the model returns all hidden-states.
    
    
    '''
        desc = removing from the text emoji,hashtag,url address and so on
    '''
    def preprocessing(self,lst_of_text):
        for i in  range(len(lst_of_text)):
            if lst_of_text[i] is not None:
                  lst_of_text[i] = str(lst_of_text[i])
                  lst_of_text[i] = re.sub(r'#([^ ]*)', r'\1',   lst_of_text[i])
                  # delete the "#" mark from beginging of the sentece.
                  lst_of_text[i] = re.sub(r'https.*[^ ]', 'URL',   lst_of_text[i])
                  lst_of_text[i]= re.sub(r'http.*[^ ]', 'URL',    lst_of_text[i])
                  lst_of_text[i] = re.sub(r'@([^ ]*)', '@USER',    lst_of_text[i])
                  lst_of_text[i] = emoji.demojize(   lst_of_text[i])
                  lst_of_text[i] = re.sub(r'(:.*?:)', r' \1 ',    lst_of_text[i])
                  lst_of_text[i] = re.sub(' +', ' ',    lst_of_text[i])
        
        return lst_of_text
            
    '''
        parameters : list of sentenes, list of labels
       # desc= tokinezed the texts of the dataset 
       #    Return the list of encoded sentences, the list of labels and the list of attention masks
    '''
    def load_data_and_tokenized(self,sents,labels):
        
        num_of_words=100
        
        #preprocessing
        sents=self.preprocessing(sents)
        
        # tokenization
        input_ids= []

        for sent in sents:
            encoded_sent = self.tokenizer.encode(
                sent,              
                add_special_tokens=True,  
                max_length=num_of_words)
                
            input_ids.append(encoded_sent)

      # # Pad our input tokens with value 0.
        input_ids = pad_sequences(input_ids, maxlen=num_of_words, dtype="long",
                        value=self.tokenizer.pad_token_id, truncating="pre", padding="pre")
                            
        # Create attention masks
        # The attention mask simply makes it explicit which tokens are actual words versus which are padding
        attention_masks = []
      
      # For each sentence in the training set
        for sent in input_ids:
            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask = [int(token_id > 0) for token_id in sent]
            # Store the attention mask for this sentence.
            attention_masks.append(att_mask)

        
        # Return the list of encoded sentences, the list of labels and the list of attention masks
        return input_ids, labels, attention_masks
        
    '''
    desc= insert the  inputs into tensors and wrap them in  dataloder in defulat batch size of 4
    '''
    def wrap_dataset(self,inputs,labels,masks,shuffle=True,batch_size = 4 ):
       #convert train inputs and labels to tensor
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        
        #attention mask
        masks = torch.tensor(masks)
    
        #wrap with dataloder 
        data = TensorDataset(inputs, masks, labels)
        sampler = RandomSampler(data)
        
        if(shuffle==True): # at training mode
            return DataLoader(data, sampler=sampler, batch_size=batch_size)
        else: #at testing mode
            return DataLoader(data, batch_size=batch_size)
    
    '''
    desc =load the dataset(s) from csv file(s)
    '''
    def load_data(self,dataset=None):
        sents=[]
        
        print(self.mode)
        if(self.mode=="training"):
            data = pd.read_csv(self.dataset1)
        else: # testing mode
           if(dataset==None):
                print("test dataset is None")
                return
           else: 
                data_test = pd.read_csv(dataset)
                data = data_test[ (data_test['label'] == 1 )] #added by chanel sub labels models
         
        if(self.topic==1):
            df = data[['topic','text','label']]
            x=list(df['topic'].values)
            y=list(df['text'].values)
            for i in range(len(y)):
                sent=x[i]+ " "+ y[i]
                sents.append(sent)
        else: # topic column not exists
            df = data[['text','label']]
            sents=list(df['text'].values)
            
        labels= list(df['label'].values)
        
        if(self.mode=="training"):
            self.train_inputs, train_labels, train_masks=self.load_data_and_tokenized(sents,labels)
            self.train_dataloader = self.wrap_dataset(self.train_inputs, train_labels, train_masks)
            
        else: # testing mode
            test_inputs, test_labels, test_masks=self.load_data_and_tokenized(sents,labels)
            test_inputs1 = test_inputs #added by chanel sub labels eval 
            shuffle=False
            batch_size=4
            self.test_dataloader=self.wrap_dataset(test_inputs, test_labels, test_masks,shuffle, batch_size)
        return sents #added by chanel sub labels eval


    '''
    desc= set model parametes
    '''
    def set_parameters(self):
        self.optimizer = AdamW(self.model.parameters(),
                  lr=1e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                )
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        self.loss_values = []
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model.to(self.device)
    
    # desc= train one epoch this function called from 'train_epochs' function 
    # rv= model loss value
    
    def train_one_epoch(self,epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        
        for i, batch in enumerate(self.train_dataloader):

            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

           
            # Make predictions for this batch
            outputs = self.model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            
                          
            # Compute the loss and its gradients
            self.loss = self.loss_fn(outputs.logits,b_labels)
            self.loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += self.loss.item()
            if i % 1000 == 999:
                last_loss = running_loss /1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.train_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    
    '''
    desc = train the model with 4 epochs
    '''
    def train_epochs(self):
        print("cuda is available",torch.cuda.is_available())
        print("torch device",self.device)
        print("model device ",self.model.device)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0
        EPOCHS = 4
        best_vloss = 1_000_000.

        for epoch in tqdm(range(0, EPOCHS), desc="Training"):
            print('EPOCH {}:'.format(epoch_number + 1))
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number,writer)
            epoch_number += 1
        
    
    def train_model(self):
        self.set_parameters()
        self.train_epochs()
        
    '''
    desc= help function to eval_model function
    '''
    def flat_accuracy(self,preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
    '''
    desc= evaluatue the model on the test dataset
    '''
    def evaluate_model(self):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        
        if(self.print_res==True):
            print("cuda is available",torch.cuda.is_available())
            print("torch device",self.device)
            print("model device ",self.model.device)
            if (self.mode=="training"): print ("the training dataset is",self.dataset1,sep="\n")
            
        #now we change the mode of the model to "testing"
        self.mode="testing"    
        for i in range(len(self.test_datasets)):
            sents = self.load_data(self.test_datasets[i])
            true_labels,pred_labels,logits=self.eval_model(self.test_dataloader)
            if np.isnan(true_labels).any():
              true_labels = np.nan_to_num(true_labels, copy=False)
            score,probs=self.sarcScore(logits)
            self.predictied_labels_for_each_ds[self.test_datasets[i]]=pred_labels
            self.true_labels_for_each_ds[self.test_datasets[i]]=true_labels
            self.tests_results[self.test_datasets[i]]=[probs,score]
            self.classification_report[self.test_datasets[i]]=classification_report(true_labels,
                                                                                pred_labels, 
                                                                                output_dict=True,digits=4,
                                                                                zero_division=0)
            
            #print f1 score and accuracy for each test dataset
            if(self.print_res==True):
                print("showing model results for model ",self.model_name,sep="\n")# added by chanel
                # print("showing model results for the test dataset : ",self.test_datasets[i],sep="\n")
                # print(classification_report(true_labels,
                #                 pred_labels, 
                #                 digits=4,zero_division=0) ) # added by chanel - comment out for hist
                print("The mean prob of the model: ",score,sep="\n") # added by chanel
                print(" ")
        return sents,[sublist[1] for sublist in probs] # add by chanel - the probs of class positive

        
        
    '''
    desc= evaluate the model on given dataloder.
          the usage of this function is for predition a full test dataset or one sentece.
        
    parameters:
        dataloder- a dataloder of test sentence or one sentece
        mission - string with 2 passable values: 'test' or 'predict'
    reutrn values:
        self.gold_labels - list of the true labels of the sentences
        self.predicted_labels - list of the model prediction labels
        self.logits_lst - list of logits of of the sentences                
        
    '''
    def eval_model(self,dataloader,mission="test"):
        self.model.eval() # turning the model to evalution mode

        # Store true lables for global eval
        self.gold_labels = []
        # Store  predicted labels for global eval
        self.predicted_labels = []
        #store logits result for all sentence
        self.logits_lst=[]
        
        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
       
        # Evaluate data for one epoch
        for batch in dataloader:
            # Add batch to GPU/CPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            if(mission=="test"):
              b_input_ids, b_input_mask, b_labels = batch
            else: # mission =="predict"
              b_input_ids, b_input_mask= batch

            with torch.no_grad():
                self.outputs = self.model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = self.outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            if(mission=="test"):
               label_ids = b_labels.to('cpu').numpy()
               #Calculate the accuracy for this batch of test sentences.
               tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
               # Accumulate the total accuracy.
               eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1
            
            '''
                the logits is matrix of results in size #(senteces in batch)* #(classes)
                in this scenario the matrix is in size 4*2
                in the variable 'pred_flat' we take the maximum in each row (axis=1) in the matrix
                so the format of 'pred_flat' is like [1,0,1,1]
            '''
            pred_flat = np.argmax(logits, axis=1).flatten()
            if(mission=="test"): labels_flat = label_ids.flatten()
            
            # Store gold labels single list
            if(mission=="test"): self.gold_labels.extend(labels_flat)
            # Store predicted labels single list
            self.predicted_labels.extend(pred_flat)
            # Store logits results for all sentence
            self.logits_lst.extend(logits)

        if(mission=="test"):    
          return self.gold_labels,self.predicted_labels ,self.logits_lst
        else:
          return self.predicted_labels ,self.logits_lst 
        
    '''
    desc= this func. called from 'sarcScore(self)' func.
          see the descripition there. 
    '''
    def calSarcMean(self,class_target):
        s=0
        for i in range(len(self.probabilities)):
            s=s+self.probabilities[i][class_target]
        return s/len(self.probabilities) 
        
    '''
    desc= Tho model outputs is tensor that for each batch in test datasets
        return the result as matrix of logits in size (batch_size*num_of_labels). for example if the batch size is 4 and the number of labels              are 2 , so the logits matrix size will be 4*2.
        
        So first we calcualte the softmax function on every row in order to recive a predection for each smaple in the range of [0,1].
        Second we, after we calcualte the "sarc score" , namely namely the avarage of the sarcasm probability along all results (namely - along             all the rows).
    '''
    def sarcScore(self,logits):
        # Compute the softmax transformation along the second axis (i.e., the rows).
          self.probabilities =[]
          for i in range(len(logits)):
            prob=softmax(logits[i])
            self.probabilities.append(prob)
          SarcScore=self.calSarcMean(1)
          return SarcScore,self.probabilities
          
    '''
        desc= return probabilities and sarcScore results as a dictionary for each test dataset
              where the key for each element in the dict is the name of the test dataset
              and the value for each element in the dict is list of two elements where
              the first elements is the probabilities for each sentence and the scoend is the sarcScore  
    '''
   
   
    '''
        desc=if the model is allready trainded, we can predict the label of given sentence.
             the function preprocess the sentence and return his predicted label
        parameters:
            sent - sentence
        return values:
            predict_label - the model prediction of the sentence
            probs - array of 2 elements , the first is the probability for non sarcastic comments and the othe for sarcastic.
    '''
    def predict(self,sent):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if(self.mode=="testing"):
            #preprocessing
            lst=[] # we create list of one element becuase func. "preprocessing"
            # is designed to received list of sentences 
            lst.append(sent) 
            preprocessed_sent_lst=self.preprocessing(lst)
            preprocessed_sent=preprocessed_sent_lst[0]
            encoded_sent = self.tokenizer.encode(
                preprocessed_sent,              
                add_special_tokens=True,  
                max_length=100)
                
            input_ids= []
            input_ids.append(encoded_sent)
            # Pad our input tokens with value 0.
            input_ids = pad_sequences(input_ids, maxlen=100, dtype="long",
                    value=self.tokenizer.pad_token_id, truncating="pre", padding="pre")
            attention_masks = []
            for sent in input_ids:
                att_mask = [int(token_id > 0) for token_id in sent]
                # Store the attention mask for this sentence.
                attention_masks.append(att_mask)
            
            #convert the inputs and the masks into tensors
            input=torch.tensor(input_ids)
            mask=torch.tensor(attention_masks)
           
            #wrap with dataloder 
            data = TensorDataset(input, mask)
            dl_sent=DataLoader(data)
            
            predict_label,logits=self.eval_model(dl_sent,"predict")
            _,probs=self.sarcScore(logits) #calculate the model probabilites of the sentence
            return predict_label[0],probs
   
    '''
        desc= save the evaluted model in the ouput folder
    '''
    def saveModel(self):
        if(self.output_model_dir!=None):
            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
                )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(self.output_model_dir)
            self.tokenizer.save_pretrained(self.output_model_dir)
            print(" model saved in ",self.output_model_dir,sep="\t")
      
