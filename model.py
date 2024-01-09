from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_model import BertModel
from model_utils import *

class Pure_Bert(nn.Module):
    '''
    BERT-SPC
    '''

    def __init__(self, args, config):
        super(Pure_Bert, self).__init__()

        self.args = args
        self.config = deepcopy(config)
        # BERT
        self.bert = BertModel.from_pretrained(args.model_dir,
                                              config=self.config)

        final_hidden_size = args.final_hidden_size
        # MLP for CLR
        mlp_layers = [
            nn.Linear(self.config.hidden_size, final_hidden_size),
            nn.ReLU()
        ]
        for _ in range(args.num_mlps - 1):
            mlp_layers += [
                nn.Linear(final_hidden_size, final_hidden_size),
                nn.ReLU()
            ]
        self.mlp = nn.Sequential(*mlp_layers)

        # classifier
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size,
                                    self.config.label_num)

    def forward(self, input_ids, token_type_ids, attention_mask):
        if self.args.spc:
            outputs = self.bert(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids)
        
        sequence_output = outputs[0]  # (N, L, D)
        pooler_output = outputs[1]  # (N, D)

        feature_output = self.mlp(sequence_output[:, 0]) # (N, D) -> (N, H)

        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output) # (batch_size, num_classes)

        return logits, feature_output
