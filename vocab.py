from corpusFunctions import tokenized_corpus


def create_vocab(file, flag):
    # regular is 0, pro is 1, semantic is 2
    editword_list = file['edit'].tolist()
    meanGrade_list = file['meanGrade'].tolist()

    if flag == 0 or flag == 1:
        corpus = file['original'].tolist()
        print('lenght of corpus is: ', len(corpus))
        print('initialising pro method by flag: ', flag)

        combined_list = []

        cleaned_corpus, vocab, combined_list = tokenized_corpus(corpus, editword_list, meanGrade_list, flag)

        edited_corpus = cleaned_corpus
        for i in range(0, len(edited_corpus)):
            if combined_list[i][0].count(' ') > 1:
                for token in edited_corpus[i]:
                    if token == combined_list[i][0] or token.lower() == combined_list[i][0]:
                        edited_corpus[i].remove(token, combined_list[i][0])
                        del edited_corpus[i][
                            edited_corpus[i].index(token):edited_corpus[i].index(token) + combined_list[i][0].count(
                                ' ')]

                        break

            else:
                for token in edited_corpus[i]:
                    if token == combined_list[i][0] or token.lower() == combined_list[i][0]:
                        edited_corpus[i][edited_corpus[i].index(token)] = combined_list[i][1]
                        # print(1)
                        break
        # print(edited_corpus)
        return cleaned_corpus, vocab, combined_list, edited_corpus