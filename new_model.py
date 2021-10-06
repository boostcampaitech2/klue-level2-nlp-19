import torch
import torch.nn as nn
from torch.nn import *
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
<<<<<<< HEAD

=======
>>>>>>> 1c08bfae067e8d9d2b6b75fd6b96d5980f8a74ea
class SimpleModel(nn.Module) :
  def __init__(self,MODEL_NAME,config):
    super().__init__()
    self.config = config
    self.klue_roberta=AutoModel.from_pretrained(MODEL_NAME,config=config, add_pooling_layer=False)
    self.num_labels = config.num_labels
    self.classifier = nn.Sequential(
      nn.Linear(config.hidden_size , config.hidden_size),
      nn.ReLU(),
      nn.Dropout(config.hidden_dropout_prob),
      nn.Linear(config.hidden_size , config.num_labels)
    )
    self.classifier.apply(self.weight_init)
    # self.init_weights()
  def weight_init(self,m):
    # classnames = m.__class__.__name__
    # if classnames.find('Linear') :
    if isinstance(m,nn.Linear):
      nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
      nn.init.zeros_(m.bias)
  def forward( self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
        Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
        config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
        If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    outputs = self.klue_roberta(            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    # print("output.shape : {0}".format(outputs.shape))
    # print(outputs)
    sequence_output = outputs[0]
    print("sequence_output.shape : {0}".format(sequence_output.shape))
    logits=self.classifier(sequence_output[:,0,:])
    loss = None
    print("logits : {0}".format(logits.shape))
    if labels is not None:
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif self.config.problem_type == "single_label_classification":

            loss_fct = CrossEntropyLoss() 
            print("self.num_labels: {0}".format(self.num_labels))
            print("logits.viewshape: {0}".format(logits.view(-1, self.num_labels).shape ) )
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
      loss=loss,
      logits=logits,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
<<<<<<< HEAD
    )
=======
    )
>>>>>>> 1c08bfae067e8d9d2b6b75fd6b96d5980f8a74ea
