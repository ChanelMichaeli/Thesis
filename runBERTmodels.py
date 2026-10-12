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
        help="set the mode of the model: 'training', 'testing', or 'affective_eval'. "
                "'training': first dataset is used for CV training, the rest for evaluation. "
                "'testing': pre-trained model, all datasets used for evaluation (aggregate metrics). "
                "'affective_eval': pre-trained model, only positive-class samples are evaluated, "
                "outputs per-sample predicted-class probabilities to scores_output_dir."
    )
    
    parser.add_argument(
        "--output_model_dir",
        default="C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/savedModels_affective/",
        #default=None, # for evaluation
        type=str,
        required=False,
        help="the a path to the folder where we save the model, if not mentioned the model will not be saved",
    )

    parser.add_argument(
        "--scores_output_dir",
        default="C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/scores_results_affective/",
        type=str,
        required=False,
        help="directory where the scores CSV is written",
    )

    parser.add_argument(
        "--scores_output_file",
        default=None,
        type=str,
        required=False,
        help="exact file path for the scores summary CSV; overrides the auto-derived path from scores_output_dir + first dataset name",
    )

    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=False,
        help="also write per-sample prediction probability CSVs to scores_output_dir (testing mode only)",
    )

    parser.add_argument(
        "--topic",
        default=0,
        type=int,
        required=False,
        help="whether to append the text column with topic column in the dataset. the default is not ",
    )

    parser.add_argument(
        "--n_folds",
        default=3,
        type=int,
        required=False,
        help="number of CV folds used during training (default: 3)",
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
                                int(args.topic),
                                n_folds=args.n_folds)

    # Route per-dataset prediction/affective-eval CSVs to scores_output_dir
    myBERTmodel.predictions_dir = args.scores_output_dir
    myBERTmodel.save_predictions = args.save_predictions

    scores = myBERTmodel.set_kFolds()

    # affective_eval writes per-sample CSVs directly; no aggregate scores to save
    if args.model_mode == "affective_eval":
        return

    # Scores summary CSV: use explicit file path if given, else derive from first dataset
    if args.scores_output_file:
        output_resutls = args.scores_output_file
    else:
        first_ds = args.datasets.split(" ")[0]
        output_resutls = os.path.join(args.scores_output_dir, first_ds)
    os.makedirs(os.path.dirname(output_resutls), exist_ok=True)

    if os.path.exists(output_resutls):
        scores.to_csv(output_resutls, mode='a', index=False, header=False)
    else:
        scores.to_csv(output_resutls, mode='w', index=False, header=True)

    if(args.model_mode == "training"):
        # Save the trained model to output_model_dir directly
        os.makedirs(args.output_model_dir, exist_ok=True)
        myBERTmodel.output_model_dir = args.output_model_dir
        myBERTmodel.saveModel()

if __name__ == "__main__":
    main()