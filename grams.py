from collections import defaultdict


def get_ngrams(sent_list, n):
    ngrams = []
    for i in range(len(sent_list) - n + 1):
        ngrams.append(sent_list[i:i + n])
    return ngrams


# return the frequency of word map that shown in corpus, removed stopwords and punctuation
def get_bigram_stat(corpus):
    print('getting bigram stat')
    bigram_freq = defaultdict(int)
    for sentence in corpus:
        # input a window size here to get bigram words
        result = get_ngrams(sentence, 1)
        # print(result)
        for bigram in result:
            bigram_freq[' '.join(bigram)] += 1
    return bigram_freq