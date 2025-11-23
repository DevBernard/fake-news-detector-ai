import torch
import torch.nn as nn
from .config import BERTIMBAU, DEVICE
from transformers import BertModel

class BaselineBertimbauClassifier(nn.Module):
    def __init__(self, 
                 pretrained_model_name: str = BERTIMBAU,
                 num_labels: int = 2,
                 **model_kwargs) -> None:
        super().__init__()
        
        self.bert = BertModel.from_pretrained(
            pretrained_model_name,
            **model_kwargs
        ).to(DEVICE)

        self.eh_ou_num_eh = nn.Linear(
            in_features=self.bert.config.hidden_size,
            out_features=num_labels
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls = outputs.last_hidden_state[:, 0, :]   # token [CLS]

        logits = self.eh_ou_num_eh(cls)

        return logits
