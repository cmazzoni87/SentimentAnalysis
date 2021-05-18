from config import get_device
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

model = T5ForConditionalGeneration.from_pretrained('t5-large')
tokenizer = T5Tokenizer.from_pretrained('t5-large')
device = get_device()
chap_n = 0


def summarize_article(token_text):
    t5_prepared_text = "summarize: " + token_text
    tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors="pt").to(device)
    summary_ids = model.to(device).generate(tokenized_text,
                                            num_beams=1,
                                            no_repeat_ngram_size=2,
                                            min_length=30,
                                            max_length=100,
                                            early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# if __name__ == '__main__':
def run_text_summarization(file_path= 'Data/Full-Economic-News'):
    financial_data = pd.read_csv('{}-Processed.csv'.format(file_path), encoding="ISO-8859-1")
    financial_data['summary'] = financial_data['text'].apply(summarize_article)
    financial_data.to_csv('{}-Summarized.csv'.format(file_path), index=False)
    return financial_data