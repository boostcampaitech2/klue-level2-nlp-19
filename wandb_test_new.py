import wandb
import json
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer



if __name__ == '__main__':
    MODEL_NAME = "klue/bert-base"
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    wandb.init(project="NLPproject", entity="chuchanghan")
    wandb.init(project="NLPproject", config=model.config)

    # with open('/opt/ml/code/results/checkpoint-40500/trainer_state.json', 'r') as f:
    #     json_data = json.load(f)
    #     # print(json_data)
    #     for e in range(5, len(json_data["log_history"]), 6):
    #         wandb.log({'epoch':json_data["log_history"][e]["epoch"] ,'eval_accuracy': json_data["log_history"][e]["eval_accuracy"], 'eval_loss': json_data["log_history"][e]["eval_loss"]})

