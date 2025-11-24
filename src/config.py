from torch import accelerator

PATH_TO_DATASET = 'documents/Fake.br-Corpus/preprocessed/pre-processed.csv'
BERTIMBAU = 'neuralmind/bert-base-portuguese-cased'
# BERTIMBAU = 'neuralmind/bert-base-portuguese-uncased' #combina com o dataset pre processado (descobri que nao existie)
SEQ_LEN = 150 #tamanho máximo da sequência de entrada
BATCH_SIZE = 16
LEARNING_RATE = 5e-5 #pra camada de classificação, deixei chumbado 1e-3
NUM_EPOCHS = 2 #o default é 3
RANDOM_SEED = 69 #( ͡° ͜ʖ ͡°)
DEVICE = accelerator.current_accelerator().type if accelerator.is_available() else "cpu" 
D_FF = 3072 #nao usei
NUM_HEADS = 12
DROPOUT = 0.1
LAYERS = 12
CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'
PAD = '[PAD]'

#nao quero que importe accelerator
__all__ = [
    'PATH_TO_DATASET',
    'BERTIMBAU',
    'SEQ_LEN',
    'BATCH_SIZE',
    'LEARNING_RATE',
    'NUM_EPOCHS',
    'RANDOM_SEED',
    'DEVICE',
    'D_FF',
    'NUM_HEADS',
    'DROPOUT',
    'LAYERS',
    'CLS',
    'SEP',
    'MASK',
    'PAD',
]
