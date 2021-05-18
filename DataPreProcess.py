import pandas as pd
import re

# clean sentences
def decontracted(text):
    # specific
    text = re.sub("won\'t", "will not", text)
    text = re.sub("can\'t", "can not", text)
    text = re.sub("n\'t", " not", text)
    text = re.sub("\'re", " are", text)
    text = re.sub("\'s", " is", text)
    text = re.sub("\'d", " would", text)
    text = re.sub("\'ll", " will", text)
    text = re.sub("\'t", " not", text)
    text = re.sub("\'ve", " have", text)
    text = re.sub("\'m", " am", text)
    return text


def replace_known_acronyms(text):
    # specific
    text = text.replace("U.S.", "United States")
    text = text.replace("U.K.", "United Kingdom")
    text = text.replace("E.U.", "European Union")
    text = text.replace("U.N.", "United Nations")
    return text


def clean_statements(sentences, deep_clean=False):
    sentences = re.sub(" '", "'", sentences)
    sentences = re.sub(" 's", "'s", sentences)
    sentences = re.sub('\( ', '(', sentences)
    sentences = re.sub(' \)', ')', sentences)
    sentences = re.sub('``', '"', sentences)
    sentences = re.sub("''", '"', sentences)
    if deep_clean:
        sentences = sentences.replace('</br>', '')
        sentences = re.sub(r'\([^)]*\)', ' ', sentences)  # removes words inside parentheses?
        sentences = decontracted(sentences)
        sentences = replace_known_acronyms(sentences)
        sentences = ' '.join([i if i.isnumeric() == False else ' NUMERIC ' for i in sentences.split() ])
        sentences = re.sub('(\d+)', '', sentences)
        sentences = re.sub ('[^A-Za-z. ]+', ' ', sentences)
        sentences = ' '.join([i for i in sentences.split() if len(i) > 1 or i == 'a'])
        sentences = sentences.replace('mn', ' ')
        sentences = re.sub(' +', ' ', sentences)
    return sentences


# if __name__ == '__main__':
def run_data_preprocessing(file_path='Data/Full-Economic-News'):
    financial_data = pd.read_csv('{}-PreProcessed.csv'.format(file_path), encoding="ISO-8859-1")
    financial_data['text'] = financial_data['raw text'].apply(clean_statements, deep_clean=True)
    financial_data = financial_data.drop_duplicates(subset='text')
    financial_data.to_csv('{}-Processed.csv'.format(file_path), index=False)
    return financial_data