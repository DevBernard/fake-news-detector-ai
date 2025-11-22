import torch
import torch.nn as nn
from config import BERTIMBAU,DEVICE
from transformers import AutoModel

class BaselineBertimbauClassifier(nn.Module):
    def __init__(self, 
                 pretrained_model_name: str = BERTIMBAU,
                 num_labels: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name).to(DEVICE) 
        self.eh_ou_num_eh = nn.Linear(
            in_features=self.bert.config.hidden_size,
            out_features=num_labels
        )

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # [batch_size, seq_len, hidden_size]
        cls: torch.Tensor = outputs.last_hidden_state[:, 0, :] #cria um plano com o token [CLS] por batch, com o tamanho do emb
        logits = self.eh_ou_num_eh(cls)

        return logits