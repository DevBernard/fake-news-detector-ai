import torch
import torch.nn as nn
from config import BERTIMBAU
from transformers import AutoModel


class CustomBertimbauClassifier(nn.Module):
    def __init__(self,
                 pretrained_model_name: str = BERTIMBAU,
                 num_labels: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(
            in_features=self.bert.config.hidden_size * 2,
            out_features=num_labels
        )

    def mean_pooling(self,
                     token_embeddings: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_emb = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9) #está pegando a média só dos tokens que não são padding

        return sum_emb / sum_mask

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # [batch_size, seq_len, hidden_size]
        lhs = outputs.last_hidden_state
        cls: torch.Tensor = lhs[:, 0, :] #cria um plano com o token [CLS] por batch, com o tamanho do emb

        mean   = self.mean_pooling(lhs, attention_mask)
        concat = torch.cat([cls, mean], dim=1)
        logits = self.classifier(concat)

        return logits #[batch_size, num_labels] um fake ou true para cada batch
