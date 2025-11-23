import inspect
from transformers import (
    BertTokenizerFast,
    Trainer, TrainingArguments
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
        elif (not self.tokenizer): #nada especifico por retornar Any
            raise ValueError("O tokenizer deve ser fornecido.")

        self._trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,

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




# args = build_training_arguments(
#     output_dir="bertimbau-cls-ptbr",
#     evaluation_strategy="epoch",
#     save_strategy="no",
#     learning_rate=2e-5,
#     weight_decay=0.01,
#     per_device_train_batch_size=BATCH,
#     per_device_eval_batch_size=BATCH,
#     num_train_epochs=EPOCHS,
#     fp16=torch.cuda.is_available(),
#     logging_steps=50,
#     report_to="none",
#     seed=SEED
# )


# print("\n=== Iniciando fine-tuning do BERTimbau ===")
# trainer.train()
# eval_out = trainer.evaluate()
# print("\nResultados de validação:", eval_out)