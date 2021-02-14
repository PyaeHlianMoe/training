from enum import Enum
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig


MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig)
}

class ModelConfig(Enum):
    MODEL_TYPE = 'bert'
    PRETRAINED_MODEL = 'bert-base-uncased'


class TrainingConfig(Enum):
    SEED = 42
    BATCH_SIZE = 4
    CLASSES = ["positive", "negative"]
    NUM_HIDDEN_LAYERS = 12
    NUM_ATTENTION_HEADS = 12
    AVAILABLE_LAYERS = 12
