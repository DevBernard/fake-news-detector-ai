# deprecated, mas vou manter por enquanto
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from .config import BERTIMBAU, DEVICE

#era mais complexo antes, até eu entender que estava fazendo retrabalho. 
# Agora virou um wrapper para manter o pardrão
class BaselineBertimbauClassifier(nn.Module):
    def __init__(self,
                 pretrained_model_name: str = BERTIMBAU,
                 num_labels: int = 2,
                 **model_kwargs) -> None:
        super().__init__()

        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            **model_kwargs
        ).to(DEVICE)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
