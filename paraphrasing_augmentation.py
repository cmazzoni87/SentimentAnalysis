from config import ModelParamsConfig as var
from config import get_device
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import pandas as pd
from tqdm import tqdm
from typing import List

tqdm.pandas()
torch_device = get_device()


def get_response(input_text, num_return_sequences, tokenizer, model) -> List[str]:


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


def execute_pegasus_augmentation(data, file_path) -> pd.DataFrame:
    MODEL_NAME = var.PARAPHRASING_MODEL
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(torch_device)
    train = data.copy()
    train = train[['summary', 'sentiment']]
    number_sequences = 10
    train['paraphrased text'] = train['summary'].progress_apply(get_response,
                                                                     num_return_sequences=number_sequences,
                                                                     tokenizer=tokenizer,
                                                                     model=model)
    generated = train.explode('paraphrased text')
    generated = generated.dropna()
    generated.to_csv('{}-Processed-Summarized-Augmented.csv'.format(file_path), index=False)
    return generated