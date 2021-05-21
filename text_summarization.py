from config import ModelParamsConfig as var
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from tqdm import tqdm

tqdm.pandas()



def summarize_article(token_text, tokenizer, model) -> str:
    device = 'cpu'
    t5_prepared_text = "summarize: " + token_text
    tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors="pt", truncation=True).to(device)
    summary_ids = model.to(device).generate(tokenized_text,
                                            num_beams=1,
                                            no_repeat_ngram_size=2,
                                            min_length=30,
                                            max_length=60,
                                            early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def run_text_summarization(data, file_path) -> pd.DataFrame:
    model = T5ForConditionalGeneration.from_pretrained(var.SUMMARIZATION_MODEL)
    tokenizer = T5Tokenizer.from_pretrained(var.SUMMARIZATION_MODEL)
    data_source = data.copy()
    data_source['summary'] = data_source['text'].progress_apply(summarize_article, tokenizer=tokenizer, model=model)
    data_source.to_csv('{}-Processed-Summarized.csv'.format(file_path), index=False)
    return data_source
