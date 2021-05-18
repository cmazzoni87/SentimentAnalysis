import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


MODEL_NAME = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda'
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(torch_device)


def get_response(input_text, num_return_sequences):
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


# if __name__ == '__main__':
def execute_pegasus_augmentation(file_path= 'Data/Full-Economic-News'):
    train = pd.read_csv('{}-Processed-Summarized.csv'.format(file_path))
    train = train[['summary', 'sentiment']]
    number_sequences = 10
    train['Paraphrased Headlines'] = train['summary'].apply(get_response, num_return_sequences=number_sequences)
    generated = train.explode('Paraphrased Headlines')
    generated = generated.dropna()
    generated.to_csv('Data/Full-Economic-News-Augmented.csv', index=False)
    return generated