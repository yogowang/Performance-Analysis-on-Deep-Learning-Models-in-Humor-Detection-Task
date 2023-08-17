import re
import string

import nltk
import pandas as pd
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import wordnet, stopwords

#nltk.download('stopwords')
#nltk.download('wordnet')
import grams
stop_words_list = stopwords.words('english')
additional_punctuation = '“”—‘’</?``' + '\'s' + '--'
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def token_processing(token):
    if token not in string.punctuation and token not in additional_punctuation:
        if '/' in token:
            result = token.replace('/', '')
            return result
        result = token
        return result


def corpus_trucasing(corpus, flag):
    country_list = pd.read_csv('data/address.csv')['country'].tolist()
    capital_list = pd.read_csv('data/address.csv')['city'].tolist()
    people_list = pd.read_csv('data/address.csv')['people'].tolist()
    dir_list = pd.read_csv('data/address.csv')['direction'].tolist()
    wnl = WordNetLemmatizer()
    vocab = []
    trucased_corpus = []

    bigram_freq = grams.get_bigram_stat(corpus)
    print('bigram_freq obtained')
    print('processing... please wait')
    for sentence in corpus:
        sentence = nltk.pos_tag(sentence)
        clean_sentence = []
        # print(sentence)
        for token, token_pos in sentence:
            wordnet_pos = get_wordnet_pos(token_pos) or wordnet.NOUN
            if flag == 1:
                if re.search(r'\d', token):
                    clean_sentence.append('number')
                    if 'number' not in vocab:
                        vocab.append('number')
                    continue
                if token.upper() in country_list or token.upper() in capital_list:
                    clean_sentence.append('address')
                    if 'address' not in vocab:
                        vocab.append('address')
                    continue
                if token.upper() in people_list:
                    clean_sentence.append('people')
                    if 'people' not in vocab:
                        vocab.append('people')
                    continue
                if token.upper() in dir_list:
                    clean_sentence.append('direction')
                    if 'direction' not in vocab:
                        vocab.append('direction')
                    continue

            if (token_pos != "NNP" and token_pos != "JJ" and token.lower() not in stop_words_list) or (bigram_freq[
                                                                                                           token] < 150 and token.lower() not in stop_words_list):
                clean_sentence.append(wnl.lemmatize(token.lower(), wordnet_pos))
                if wnl.lemmatize(token.lower(), wordnet_pos) not in vocab:
                    vocab.append(wnl.lemmatize(token.lower(), wordnet_pos))
                continue
                # print(clean_sentence)
            else:
                clean_sentence.append(wnl.lemmatize(token, wordnet_pos))
                if wnl.lemmatize(token.lower(), wordnet_pos) not in vocab:
                    vocab.append(wnl.lemmatize(token.lower(), wordnet_pos))
        trucased_corpus.append(clean_sentence)
    return trucased_corpus, vocab


def corpus_roughclean(corpus, editword_list, meanGrade_list, flag):
    cleaned_corpus = []
    extractword_list = []
    combined_list = []
    for sentence in corpus:
        # extract old word
        extractword_list.append(sentence[sentence.index('<') + 1:sentence.index('/')])

        clean_sentence = []
        # tokenizing sentence
        sentence = word_tokenize(sentence)

        # get pos tag of words
        sentence = nltk.pos_tag(sentence)
        for token, token_pos in sentence:
            # remove stopwords adn punctuation
            if token.lower() not in stop_words_list:
                token = token_processing(token)
                if token is not None:
                    clean_sentence.append(token)
        cleaned_corpus.append(clean_sentence)

    for i in range(0, len(corpus)):
        combined_list.append([extractword_list[i], editword_list[i]])
    # lowering words in combined list
    # print(combined_list)
    test, _ = corpus_trucasing(combined_list, flag)
    # print(len(test))
    combined_list = []
    # combined_list: 0:old word, 1: new old, 2: meanGrade
    for i in range(0, len(corpus)):
        combined_list.append([test[i][0], test[i][1], meanGrade_list[i]])
    print('returned combined_list: 0:old word, 1: new old, 2: meanGrade')
    return cleaned_corpus, combined_list


# truecasing the corpus
def tokenized_corpus(corpus, editword_list, meanGrade_list, flag):
    print()
    print('tokenized_corpus initialising')
    vocab = []
    mid_corpus = []
    final_corpus = []

    print()
    print('corpus 1st cleaning starting: tokenizing, stop words removed, punctuation removed, and number labelled')
    print('please wait')
    # combined_list: 0:old word, 1: new old, 2: meanGrade
    mid_corpus, combined_list = corpus_roughclean(corpus, editword_list, meanGrade_list, flag)
    print('corpus 1st cleaning is completed: tokenizing, stop words removed, punctuation removed, and number labelled')
    print()
    print('corpus 2ec round starting: address labelled, direction labelled, and people labelled')
    print('please wait')
    final_corpus, vocab = corpus_trucasing(mid_corpus, flag)
    print('corpus 2ec round completed: address labelled, direction labelled, and people labelled')
    # print(final_corpus)
    print('corpus and vocab returned')

    return final_corpus, vocab, combined_list