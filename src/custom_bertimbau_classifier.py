import torch
import torch.nn as nn
from config import BERTIMBAU, DEVICE
from transformers import BertForSequenceClassification as BFSC, SpecificPreTrainedModel as ModelType


class CustomBertimbauClassifier(nn.Module):
    def __init__(self,
                 pretrained_model_name: str = BERTIMBAU,
                 num_labels: int = 2,
                 **model_kwargs) -> None:
        super().__init__()
        self.bert: ModelType  = BFSC.from_pretrained(pretrained_model_name, **model_kwargs).to(DEVICE) #vou colocar fora também para garantir
        #simplesmente nao tem como fazer o bert mostrar seus métodos...
        self.eh_vdd_ou_nao = nn.Linear(
            in_features=self.bert.config.hidden_size * 2,
            out_features=num_labels
        )

    def _expand_one_dimension(self,tensor: torch.Tensor, dim_size: torch.Size) -> torch.Tensor:
        return tensor.unsqueeze(-1).expand(dim_size)

    def _apply_padding_mask(self,
                     token_embeddings: torch.Tensor,
                     exp_attn_mask: torch.Tensor) -> torch.Tensor:
            #exp_attn mask é 1 pros tokens que importam e 0 pros pads
            #exp_attn_mask: [batch_size, seq_len, hidden_size]
            
        return token_embeddings.masked_fill(exp_attn_mask == 0, 0.0)

    def mean_pooling(self,
                     token_embeddings: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:

        hidden_size: torch.Size = token_embeddings.size() 
        expandend_mask: torch.Tensor = self._expand_one_dimension(attention_mask, hidden_size)
        expandend_mask = expandend_mask.to(device=DEVICE,
                                            dtype=torch.float32)

        no_pad_emb = self._apply_padding_mask(token_embeddings, expandend_mask)
        sum_emb = torch.sum(no_pad_emb, dim=1)
        sum_mask = torch.clamp(expandend_mask.sum(dim=1), min=1e-9) #está pegando a média só dos tokens que não são padding

        return sum_emb / sum_mask

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # [batch_size, seq_len, hidden_size]
        lhs = outputs.last_hidden_state
        cls: torch.Tensor = lhs[:, 0, :] #cria um plano com o token [CLS] por batch, com o tamanho do emb

        mean   = self.mean_pooling(lhs, attention_mask)
        concat = torch.cat([cls, mean], dim=1)
        logits = self.eh_vdd_ou_nao(concat)

        return logits #[batch_size, num_labels] um fake ou true para cada batch (ainda precisa de softmax)