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


def clean_statements(sentences, deep_clean=False):
    sentences = re.sub(" '", "'", sentences)
    sentences = re.sub(" 's", "'s", sentences)
    sentences = re.sub('\( ', '(', sentences)
    sentences = re.sub(' \)', ')', sentences)
    sentences = re.sub('``', '"', sentences)
    sentences = re.sub("''", '"', sentences)
    # sentences = re.sub('\s([?.,%:!"](?:\s|$))', r'\1', sentences)
    if deep_clean:
        sentences = re.sub(r'\([^)]*\)', ' ', sentences)  # removes words inside parentheses?
        sentences = decontracted(sentences)
        # sentences = re.sub('[^A-Za-z0-9 ]+', '', sentences)
        sentences = ' '.join([i if i.isnumeric() == False else ' NUMERIC ' for i in sentences.split() ])
        sentences = re.sub('(\d+)', '', sentences)
        sentences = re.sub('[^A-Za-z ]+', '', sentences)
        sentences = ' '.join([i for i in sentences.split() if len(i) > 1 or i == 'a'])
        sentences = sentences.replace('mn', ' ')
        sentences = re.sub(' +', ' ', sentences)

    return sentences


financial_data = pd.read_csv('Data/hybrid_curated_data.csv', encoding="ISO-8859-1")
financial_data['Headline'] = financial_data['Headline'].apply(clean_statements, deep_clean=True)
financial_data.to_csv('Data/hybrid_deep_curated_data.csv', index=False)