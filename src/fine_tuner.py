import inspect
from torch import optim
from transformers import (
    BertTokenizerFast,
    Trainer, TrainingArguments,
    get_linear_schedule_with_warmup
)

class FineTuner():
    def __init__(self, tokenizer: BertTokenizerFast = None):
        self.tokenizer = tokenizer

    def export_model(self, path: str) -> None:
        self._trainer.save_model(path)

    def train(self) -> dict:
        self._trainer.train()
        
        return self._trainer.evaluate()

    # def set_tokenizer(self, model_name) -> 'FineTuner':
    #     self.tokenizer = BertTokenizerFast.from_pretrained(model_name) #gonna be used a lot la fora
    #     return self

    def set_compute_metrics(self, function) -> 'FineTuner':
        self._compute_metrics = function
        return self


    def set_trainer_optimizer_params(self, **kwargs) -> 'FineTuner':
        custom_optimizer = optim.AdamW([
            {'params': self.model.bert.parameters(), 'lr': kwargs.get('lr_bert', 5e-5)},
            {'params': self.model.classifier.parameters(), 'lr': kwargs.get('lr_classifier', 1e-3)}
        ])
        custom_scheduler = get_linear_schedule_with_warmup(custom_optimizer, 100, 900)

        self._custom_optimizer: tuple[optim.AdamW, optim.lr_scheduler.lambdaLR] = (custom_optimizer, custom_scheduler)
        return self


    def set_trainer(
        self,
        model,
        train_dataset,
        eval_dataset,
    ) -> 'FineTuner':

        if (not hasattr(self, '_training_args')):
            raise ValueError("Defina os argumentos de treinamento antes de criar o Trainer.")
        elif (not callable(self._compute_metrics)):
            raise ValueError("A função compute_metrics deve ser fornecida e ser chamável.")
        elif (not hasattr(self, '_custom_optimizer')):
            raise ValueError("Defina os parâmetros do otimizador antes de criar o Trainer.")
        elif (not self.tokenizer): #nada especifico por retornar Any
            raise ValueError("O tokenizer deve ser fornecido.")

        self._trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,

            optimizers=self._custom_optimizer,

            compute_metrics=self._compute_metrics,
            args=self._training_args,
            tokenizer=self.tokenizer
        )
        return self

    def set_training_arguments(self, **kwargs) -> 'FineTuner':
        sig = inspect.signature(TrainingArguments.__init__)
        allowed = set(sig.parameters.keys()); allowed.discard("self")
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        dropped = [k for k in kwargs if k not in allowed]
        if dropped:
            print("[Aviso] parâmetros ignorados nesta versão:", dropped)
        
        self._training_args = TrainingArguments(**filtered)

        return self