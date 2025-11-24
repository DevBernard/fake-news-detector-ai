from torch import optim as torch_optim
from torch import tensor as torch_tensor, long as torch_long
from torch.utils.data import Dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast as TokenizerType, BertTokenizerFast
from .config import RANDOM_SEED

class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self._labels = labels
    
    def __len__(self):
        return len(self._labels)
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch_tensor(self._labels[idx], dtype=torch_long)
        return item

class DatasetLoader():
    def __init__(self, path, model_name, max_len):
        self._model_name = model_name
        self.path = path #caminho do arquivo ja preprocessado
        self.max_len = max_len

    
    def __len__(self):
        return len(self._texts)


    def load_dataset(self, seed: int = RANDOM_SEED):
        ( self
            ._set_tokenizer(self._model_name)
            ._set_texts_and_labels()
            ._set_labels_mapping()
            ._set_train_val_test(seed))
        
        X_train_enc = self._tokenize_batch(self._X_train)
        X_val_enc   = self._tokenize_batch(self._X_val)
        X_test_enc  = self._tokenize_batch(self._X_test)

        self._train_dataset = FakeNewsDataset(X_train_enc, self._y_train)
        self._val_dataset   = FakeNewsDataset(X_val_enc, self._y_val)
        self._test_dataset  = FakeNewsDataset(X_test_enc, self._y_test)

        return self

    def _tokenize_batch(self, texts):
        return self.tokenizer(
            texts, #this texts are the batch of texts from X_train, X_val or X_test
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
    
    def _set_tokenizer(self, model_name: str) -> 'DatasetLoader': #bertimbau tokenizer
        self.tokenizer: TokenizerType = BertTokenizerFast.from_pretrained(model_name)
        return self
    
    def _set_texts_and_labels(self) -> 'DatasetLoader':
        df = read_csv(self.path)
        self._texts: list[str] = df['preprocessed_news'].tolist()
        self._labels: list[str] = df['label'].tolist()

        return self

    def _set_labels_mapping(self) -> 'DatasetLoader':
        unique_labels = set(self._labels)
        self.label2id: list[int] = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label: list[str] = {idx: label for label, idx in self.label2id.items()}
        self.num_labels = len(unique_labels)

        return self
    
    def _set_trainer_optmizer_params(self, **kwargs) -> 'DatasetLoader':

        return self

    def _set_train_val_test(self, seed: int) -> 'DatasetLoader':
        labels_in_ids: list[int] = [self.label2id[label] for label in self._labels]

        self._X_train, X_temp, self._y_train, y_temp = train_test_split( #pega 70% dos dados pro treino
            self._texts, labels_in_ids, test_size=0.3, random_state=seed, stratify=labels_in_ids
        )
        self._X_val, self._X_test, self._y_val, self._y_test = train_test_split( #pega os 30% restantes e divide entre validação e teste
            X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
        )

        return self

    def get_tokenizer(self) -> TokenizerType:
        if not hasattr(self, 'tokenizer'):
            raise ValueError("O tokenizer ainda não foi definido. Carregue o dataset primeiro.")
        return self.tokenizer

    def get_texts_with_labels(self, path: str) -> tuple[list[str], list[str]]:
        return self._texts, self._labels

    def get_labels_mapping(self) -> dict:
        """Retorna um dicionário com as chaves 'label2id', 'id2label' e 'num_labels'."""
        if (not hasattr(self, 'label2id') or
            not hasattr(self, 'id2label') or
            not hasattr(self, 'num_labels')):
            raise ValueError("O mapeamento de labels ainda não foi definido. Carregue o dataset primeiro.")
        #bate com os kwargs do BertForSequenceClassification
        return {'label2id': self.label2id, 'id2label': self.id2label, 'num_labels': self.num_labels}

    def get_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        if (not hasattr(self, '_train_dataset') or
            not hasattr(self, '_val_dataset') or
            not hasattr(self, '_test_dataset')):
            raise ValueError("Os datasets ainda não foram definidos. Carregue o dataset primeiro.")
        
        return self._train_dataset, self._val_dataset, self._test_dataset