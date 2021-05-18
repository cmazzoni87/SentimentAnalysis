from config import ModelParamsConfig as var
from config import get_device
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import pandas as pd
from tqdm import tqdm
from typing import List

tqdm.pandas()


def get_response(input_text, num_return_sequences) -> List[str]:
    MODEL_NAME = var.PARAPHRASING_MODEL
    torch_device = get_device()
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(torch_device)
    batch = tokenizer.prepare_seq2seq_batch([input_text],
                                            truncation=True,
                                            padding='longest',
                                            return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,
                                num_beams=num_return_sequences,
                                num_return_sequences=num_return_sequences,
                                temperature=1.5).to(torch_device)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


def execute_pegasus_augmentation(data) -> pd.DataFrame:
    train = data.copy()
    train = train[['summary', 'sentiment']]
    number_sequences = 10
    train['paraphrased headlines'] = train['summary'].progress_apply(get_response, num_return_sequences=number_sequences)
    generated = train.explode('paraphrased headlines')
    generated = generated.dropna()
    generated.to_csv('Data/Economic-News-Processed-Summarized-Augmented.csv', index=False)
    return generated