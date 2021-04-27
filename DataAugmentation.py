import pandas as pd
import nltk
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import requests
from transformers import pipeline
from DataPreProcess import clean_statements


tqdm.pandas()
url = 'http://paraphrase.org/api/en/search/'
stoplist = stopwords.words('english')


def create_pipeline(model_name):
   return pipeline('fill-mask', model=model_name)


def get_synonyms(word):
    results = []
    querystring = {'batchNumber': '0', 'needsPOSList': 'true', 'q': word}
    headers = {'cache-control': 'no-cache', 'postman-token': '2d3d31e7-b571-f4ae-d69b-8116ff64d752'}
    response = requests.request('GET', url, headers=headers, params=querystring)
    response_js = response.json()
    res_count = response_js['hits']['found']
    if res_count > 0:
        res_count = min(10, res_count)
        hits = response_js['hits']['hit'][:res_count]
        results = [hit['target'] for hit in hits]
    return results


def mask_words(text):
    new_text = ' '.join([i for i in text.split() if i not in stoplist
                         and any(map(str.isupper, i)) is False
                         and any(map(str.isnumeric, i)) is False])
    words = word_tagger(new_text)
    words = [i[0] for i in words if i[1][0] == 'V']
    masked_sentences = []
    for word in words:
        masked_sentence = text.replace(' {} '.format(word), ' [MASK] ', 1)
        if '[MASK]' not in masked_sentence:
            masked_sentence = text.replace('{} '.format(word), '[MASK] ', 1)
        if '[MASK]' not in masked_sentence:
            masked_sentence = text.replace(' {}'.format(word), ' [MASK]', 1)
        if '[MASK]' not in masked_sentence:
            print('blaa  again')
        masked_sentences.append(masked_sentence)
    return masked_sentences


def treebankTag(text):
    words = nltk.word_tokenize(text)
    treebankTagger = nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle')
    return treebankTagger.tag(words)


def word_tagger(text):
    return nltk.pos_tag(word_tokenize(text))


def find_synonym(text, sentiment):
    texts = []
    texts.append(text)
    word_tags = word_tagger(text)
    verbs = [i[0] for i in word_tags if i[1][0] == 'V' or i[1] == 'RB']
    to_exchange = verbs
    try:
        for word in to_exchange:
            similar_words = get_synonyms(word)
            for similar in similar_words:
                texts.append(re.sub(word, similar, text))
    except Exception as e:
        print(e)
    return texts, sentiment


if __name__ == '__main__':
    # 'dw_crowdflower_news_data_clean'
    # 'labeled_headlines_labeled'
    train = pd.read_csv('Data/hybrid_deep_curated_data.csv')
    train = train[['Headline', 'Sentiment']]
    train['Headline'] = train['Headline'].apply(clean_statements, deep_clean=True)
    predict_mask = create_pipeline('bert-base-cased')
    generated = train
    generated['Masked Headlines'] = generated['Headline'].apply(mask_words)
    generated = generated.explode('Masked Headlines')
    generated = generated.dropna()
    generated['Augmented'] = generated['Masked Headlines'].apply(predict_mask)
    generated = generated.explode('Augmented')
    generated = generated.reset_index(drop=True)
    augmented = generated['Augmented'].apply(pd.Series)
    augmented = augmented.reset_index(drop=True)
    results = generated.join(augmented)
    results['Headline'] = results['sequence']
    # results = results.drop_duplicates()
    mask = results['token_str'].str.match('[a-zA-Z]')
    results = results[mask]
    results.to_csv('headlines_labeled_augmented.csv', index=False)
    # print(augmented)
    # augmeted_text = train.progress_apply(lambda x : find_synonym(x['Headline'], x['Binary Sentiment']), axis=1)
    # x,y = list(map(list,zip(*augmeted_text.values.tolist())))
    # new_df = pd.DataFrame({'Augmented Text': x, 'Sentiment': y})
    # new_df = new_df.explode('Augmented Text')
    # new_df.to_csv('labeled_headlines_augmented.csv',index=False)
