from torch import accelerator
BERTIMBAU = 'neuralmind/bert-base-portuguese-cased'
SEQ_LEN = 512
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
RANDOM_SEED = 69 #( ͡° ͜ʖ ͡°)
DEVICE = accelerator.device() if accelerator.is_available() else 'cpu'
D_FF = 3072
NUM_HEADS = 12
DROPOUT = 0.1
LAYERS = 12
CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'
PAD = '[PAD]'

#nao quero que importe accelerator
__all__ = [
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
