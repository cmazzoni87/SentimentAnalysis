import pandas as pd
import re
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.corpus import wordnet


tqdm.pandas()
stop_words = stopwords.words('english')
NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]



def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if word != l.name():
                synonyms.append(l.name())
    return synonyms


def find_synonym(text, sentiment):
    sentences = [text]
    texts = text.split()
    texts = texts[:len(texts) // 2]
    word_bag = [i for i in texts if i not in stop_words
                and any(map(str.isupper, i)) is False
                and i not in NUMBERS
                and i != 'NUMERIC']
    word_bag = list(set(word_bag))
    n = 2
    combined_bag = [word_bag[i * n:(i + 1) * n] for i in range((len(word_bag) + n - 1) // n)]
    to_exchange = combined_bag
    payload = text
    try:
        for words in to_exchange:
            for word in words:
                similar_words = get_synonyms(word)
                if similar_words is not None:
                    similar_words = [re.sub('[^A-Za-z ]+', ' ', sent) for sent in similar_words]
                    for similar in similar_words:
                        payload = re.sub(word, similar, payload)
                    sentences.append(payload)
        sentences = list(set(sentences))
    except Exception as e:
        print(e)
    return sentences, sentiment


# if __name__ == '__main__':
def execute_synonym_replacement(file_path= 'Data/Full-Economic-News'):
    train = pd.read_csv('{}-Augmented.csv'.format(file_path))
    train = train[['sentiment', 'paraphrased text']]
    augmeted_text = train.progress_apply(lambda x : find_synonym(x['paraphrased text'], x['sentiment']), axis=1)
    x,y = list(map(list,zip(*augmeted_text.values.tolist())))
    new_df = pd.DataFrame({'generated text': x, 'sentiment': y})
    new_df = new_df.explode('generated text')
    new_df = new_df.dropna()
    new_df = new_df.drop_duplicates()
    new_df.to_csv('{}-Syn-Replaced.csv'.format(file_path),index=False)
    return new_df
