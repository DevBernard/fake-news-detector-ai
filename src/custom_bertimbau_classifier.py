import torch
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from .config import BERTIMBAU, DEVICE


class CustomBertimbauClassifier(nn.Module):
    def __init__(self,
                 pretrained_model_name: str = BERTIMBAU,
                 num_labels: int = 2,
                 **model_kwargs) -> None:
        super().__init__()

        # Usamos BertModel, porque queremos os hidden states e pooling custom
        self.bert = BertModel.from_pretrained(
            pretrained_model_name,
            **model_kwargs
        ).to(DEVICE)

        hidden = self.bert.config.hidden_size

        # concat CLS (hidden) + mean pooling (hidden) => 2 * hidden => 2 * 768 = 1536
        self.classifier = nn.Linear(
            in_features=hidden * 2,
            out_features=num_labels
        )


    def _expand_one_dimension(self, tensor: torch.Tensor, dim_size: torch.Size) -> torch.Tensor:
        # attention_mask: [batch, seq] → [batch, seq, hidden]
        return tensor.unsqueeze(-1).expand(dim_size) #ainda precisa de squeeze será

    def _apply_padding_mask(self,
                            token_embeddings: torch.Tensor,
                            exp_attn_mask: torch.Tensor) -> torch.Tensor:
        return token_embeddings.masked_fill(exp_attn_mask == 0, 0.0)

    def mean_pooling(self,
                     token_embeddings: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:

        # token_embeddings: [batch, seq_len, hidden_size]
        # attention_mask:   [batch, seq_len]
        

        batch, seq_len, hidden = token_embeddings.size()
        expanded_mask = self._expand_one_dimension(attention_mask, (batch, seq_len, hidden))

        expanded_mask = expanded_mask.to(device=DEVICE, dtype=torch.float32)

        masked_emb = self._apply_padding_mask(token_embeddings, expanded_mask)

        sum_emb = masked_emb.sum(dim=1) # soma só onde mask=1
        sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)

        return sum_emb / sum_mask # média sem pad

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor | None) -> SequenceClassifierOutput:

        # saída do BertModel: tem last_hidden_state sempre
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        lhs = outputs.last_hidden_state # [batch, seq, hidden]
        cls = lhs[:, 0, :] # token CLS

        mean = self.mean_pooling(lhs, attention_mask)
        concat = torch.cat([cls, mean], dim=1) # [batch, hidden*2]

        logits = self.classifier(concat)


        #adicionando uma loss manual, pois apenas a assim o Trainer reconhece
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            logit_saneado = logits.view(-1, self.classifier.out_features) #[batch*seq, 2]
            label_saneado = labels.view(-1) #[batch*seq]
            loss = loss_fn(logit_saneado, label_saneado) #[batch*seq]


        return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
            ) 
