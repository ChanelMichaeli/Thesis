import argparse
# from  bertKfold import *
# from bertModelTest import *
# from bertModel import *
from bertFineTuneKfolds import *

def parse_args():
    """
    Parse command line arguments.
    :return: the parsered arguments
    """
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument(
        "--absolute_path",
        # default="C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/Datasets/RoastMeDatasets/", # chanel - use for sublabels train
        default="C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/LLM/datasets/",
        type=str,
        required=False,
        help="the absolute path for datasets",
    )
    
    parser.add_argument(
        "--datasets",
        default="dataset/",
        type=str,
        required=False,
        help="the paths to datasets where the first is using for training and the else for tests",
    )
    
   
    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",#"sentence-transformers/stsb-bert-base"
        type=str,
        required=False,
        help="for the completed list of models names, please look in https://huggingface.co/models",
    )
    
    parser.add_argument(
        "--model_mode",
        default= "training", # chanel for evaluation: "testing",
        type=str,
        required=False,
        help="set the mode of the model: training or testing. If 'training' is chosen so the" 
                +"first dataset use for train and the rest for test. If 'test' is chosen so" 
                +" the model is pre-trained and all the datasets will be use for testing"
    )
    
    parser.add_argument(
        "--output_model_dir",
        default="C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/savedModels_finetuned/", # chanel for model
        #default=None, # for evaluation
        type=str,
        required=False,
        help="the a path to the folder where we save the model, if not mention the model will not be saved ",
    )
    
    parser.add_argument(
        "--topic",
        default=0,
        type=int,
        required=False,
        help="whether to append the text column with topic column in the dataset. the default is not ",
    )

    args = parser.parse_args()
    return args
    
############################
# for testing only!!
# def main():
#     models_path = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/savedModels_aux/"
#     # models_lst = [models_path + "humorBERT",models_path + "ironyBERT",models_path + "politenessBERT",models_path + "toxicityBERT",models_path + "empathyBERT",models_path + "offensiveBERT" ]
#     # model_names = ["humorBERT","ironyBERT","politenessBERT","toxicityBERT","empathyBERT","offensiveBERT"]
#     models_lst = [models_path + "humorBERT",models_path + "ironyBERT",models_path + "toxicityBERT"]
#     model_names = ["humorBERT","ironyBERT","toxicityBERT"]
#     args = parse_args()
#     data_results = pd.DataFrame()
#     print("my args",args,sep="\n")
#     for i in range(len(models_lst)):
#         myBERTmodel=BERTModelTest(models_lst[i],
#                                 args.absolute_path,
#                                 args.datasets,
#                                 args.output_model_dir,
#                                 "testing",
#                                 int(args.topic))
#         sents, probs= myBERTmodel.evaluate_model()
#         data_results[model_names[i]] = probs
#     data_results["text"] = sents 
#     # output_resutls = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/roastMe_eval/Train_positive_datasets_probs/" + args.datasets.split("/")[0] + ".csv"
#     output_resutls = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/Datasets/toxicity_downsample/toxicity_probs/" + args.datasets.split("/")[1]# + ".csv"
#     data_results.to_csv(output_resutls,index=False, header=True)
#####################

# def main():
#     args = parse_args()
#     print("my args",args,sep="\n")
#     myBERTmodel=BERTModel(args.model_name,
#                                 args.absolute_path,
#                                 args.datasets,
#                                 args.output_model_dir,
#                                 args.model_mode,
#                                 int(args.topic))
#     myBERTmodel.load_data()
#     myBERTmodel.train_model()
#     myBERTmodel.evaluate_model()
#     myBERTmodel.saveModel()
    
#     scores_res = pd.DataFrame([(key, *values) for key, values in myBERTmodel.get_all_scores().items()], columns=['test data', 'precision','recall','f1 score'])
#     # f1_results = pd.DataFrame(list(myBERTmodel.get_f1_score().items()), columns=['test data', 'f1 score'])
#     output_resutls = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/scores_results/app2/" + args.datasets.split(" ")[0]
#     # output_resutls = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/f1_results2/" + args.datasets.split(" ")[0]
#     # f1_results.to_csv(output_resutls,index=False, header=True)
#     scores_res.to_csv(output_resutls,index=False, header=True)


import os
def main():
    args = parse_args()
    print("my args",args,sep="\n")
    myBERTmodel=BERTKfold(args.model_name,
                                args.absolute_path,
                                args.datasets,
                                args.output_model_dir,
                                args.model_mode,
                                int(args.topic))
    scores = myBERTmodel.set_kFolds()
    # output_resutls = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/scores_results_sampled_all/" + args.datasets.split(" ")[0]
    # output_resutls = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/scores_results_finetuned/aux_labels/negative_labels/" + args.datasets.split(" ")[0]
    # output_resutls = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/scores_results_finetuned/" + args.datasets.split(" ")[0]
    # output_resutls = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/score_results_stsb_bert_uncased/" + args.datasets.split(" ")[0]
    output_resutls = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/scores_results_upsampling_neg/" + args.datasets.split(" ")[0]
    # output_resutls = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/roastMe_eval/" + args.datasets.split(" ")[0]

    if(args.model_mode == "training"):
        
        if os.path.exists(output_resutls):
            # Append data without headers
            scores.to_csv(output_resutls, mode='a', index=False, header=False)
        else:
            # Write data with headers
            scores.to_csv(output_resutls, mode='w', index=False, header=True)
    
        # myBERTmodel.saveModel()

if __name__ == "__main__":
    main()