import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel


class CustomBertimbauClassifier ( nn.Module ):
    """
    Modelo de classificacao binaria para deteccao de fake
    news
    baseado no BERTimbau , com pooling alternativo .
    Estrategia de pooling :
    - Extrai o embedding do token [CLS ].
    - Calcula a media dos embeddings dos tokens validos .
    - Concatena [CLS ] + media e aplica uma camada de
    classificacao .
    """
    def __init__ ( self,
                   pretrained_model_name: str = "neuralmind/bert-base-portuguese-cased",
                   num_labels: int = 2):
        """
        Inicializa o modelo .
        Args :
            pretrained_model_name : nome do modelo BERTimbau
            pre - treinado .
            num_labels : numero de classes de saida (2 para
            fake / true ).
        """
        super().__init__()
        # TODO : carregar o BERTimbau e definir a cabeca de classificacao
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear( #traz uma camada linear treinavel, com pesos e bias
            in_features=self.bert.config.hidden_size * 2, #quantas features vao entrar na camada linear
            out_features=num_labels) #(Fake ou Verdadeiro)

    def mean_pooling ( self,
                       token_embeddings: torch.Tensor ,
                       attention_mask: torch.Tensor ) -> torch.Tensor:
        """
        Calcula a media dos embeddings validos ( ignora
        tokens de padding ).
        Args :
        3
        token_embeddings : tensor [ batch_size , seq_len ,
        hidden_size ]
        attention_mask : tensor [ batch_size , seq_len ]
        Returns :
        Tensor [ batch_size , hidden_size ] com a media dos
        tokens validos .
        """
        pass
    def forward ( self,
                  input_ids: torch.Tensor,
                  attention_mask: torch.Tensor ) -> torch.Tensor:
        """
        Passagem direta no modelo .
        Passos :
        1. Obtem os embeddings da ultima camada do BERT .
        2. Extrai o vetor [CLS ].
        3. Calcula a media dos embeddings validos .
        4. Concatena [CLS ] + media .
        5. Passa pelo classificador linear .
        Returns :
        logits : tensor [ batch_size , num_labels ]
        """
        pass