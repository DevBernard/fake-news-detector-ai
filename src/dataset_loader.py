from torch import tensor as torch_tensor, long as torch_long
from torch.utils.data import Dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast as TokenizerType, BertTokenizerFast
from config import RANDOM_SEED

class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch_tensor(self.labels[idx], dtype=torch_long)
        return item

class DatasetLoader():
    def __init__(self, texts, labels, tokenizer, max_len):
        self._texts = texts #ja preprocessado
        self._labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    
    def __len__(self):
        return len(self._texts)


    def load_dataset(self, path, seed: int = RANDOM_SEED):
        ( self
            ._set_tokenizer(self.tokenizer, self.max_len)
            ._set_labels_mapping(self._labels)
            ._set_texts_and_labels(path)
            ._set_train_val_test(seed))
        
        X_train_enc = self._tokenize_batch(self._X_train)
        X_val_enc   = self._tokenize_batch(self._X_val)
        X_test_enc  = self._tokenize_batch(self._X_test)

        train_dataset = FakeNewsDataset(X_train_enc, self._y_train)
        val_dataset   = FakeNewsDataset(X_val_enc, self._y_val)
        test_dataset  = FakeNewsDataset(X_test_enc, self._y_test)

        return train_dataset, val_dataset, test_dataset


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
    
    def _set_texts_and_labels(self, path: str) -> 'DatasetLoader': 
        df = read_csv(path)
        self._texts: list[str] = df['preprocessed_news'].tolist()
        self._labels: list[str] = df['label'].tolist()

        return self

    def _set_labels_mapping(self) -> 'DatasetLoader':
        unique_labels = set(self._labels)
        self.label2id: list[int] = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label: list[str] = {idx: label for label, idx in self.label2id.items()}
        self.num_labels = len(unique_labels)

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

    def get_texts_with_labels(self, path: str) -> tuple[list[str], list[str]]:
        return self._texts, self.labels

    def get_labels_mapping(self) -> dict:
        #bate com os kwargs do BertForSequenceClassification
        return {'label2id': self.label2id, 'id2label': self.id2label, 'num_labels': self.num_labels}