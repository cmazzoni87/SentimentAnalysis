from dataclasses import dataclass
import torch


def get_device() -> str:
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


@dataclass(frozen=True)
class ModelParamsConfig:
    RANDOM_SEED: int = 17
    EPOCHS: int = 3
    TRAIN_TEST_SPLIT: float = 0.3
    MODEL_NAME: str = "bert-base-uncased"
    MODEL_PATH = 'results/sentiment-analysis-with-augmented'
    SUMMARIZATION_MODEL: str = 't5-large'
    PARAPHRASING_MODEL: str = 'tuner007/pegasus_paraphrase'
    BATCH_SIZE: int = 128
    MAX_LEN: int = 60
    LR: float = 1e-4
    LABEL_NUM: int = 2


@dataclass(frozen=True)
class DataProcessorConfig:
    SOURCE_FILE = 'data/Economic-News'
    PEGASUS_SEQ = 10
    SENTIMENT_COL: str = 'sentiment'
    TEXT_COL: str = 'raw text'
