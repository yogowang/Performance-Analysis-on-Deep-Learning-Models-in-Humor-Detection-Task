import codecs
import hyperparameters
import models
import numpy as np
import pandas as pd
import nltk
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from paddings import collate_fn_padd
from task1dataset import Task1Dataset
from testEvalTrain import train1, test1,train,test
from vocab import create_vocab
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')
# Number of epochs
epochs = 10
# Proportion of training data for train compared to dev
train_proportion = 0.8
# Setting random seed and device
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
functionDic={
    'vgg':[train,test],
    'TransformerModel':[train1,test1],
    'NeuralNetwork':[train,test]
}
if __name__ == "__main__":
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    corpus = []
    vocab = []
    combined_list = []
    corpus, vocab, combined_list, edited_corpus = create_vocab(train_df, 0)
    corpus_test, vocab_test, combined_list_test, edited_corpus_test = create_vocab(test_df, 0)
    print(len(corpus), len(vocab), len(combined_list))
    wvecs = []  # word vectors
    word2idx = []  # word2index
    idx2word = []
    meangrade = []
    real = []
    with codecs.open('glove.6B.100d.txt', 'r', 'utf-8') as f:
        index = 0
        for line in tqdm(f.readlines()):
            # Ignore the first line - first line typically contains vocab, dimensionality
            if len(line.strip().split()) > 3:
                (word, vec) = (line.strip().split()[0],
                               list(map(float, line.strip().split()[1:])))

                wvecs.append(vec)
                word2idx.append((word, index))
                idx2word.append((index, word))
                index += 1
    wvecs = np.array(wvecs)
    word2idx = dict(word2idx)
    idx2word = dict(idx2word)
    vectorized_seqs = [[word2idx[tok] for tok in seq if tok in word2idx] for seq in corpus]
    # To avoid any sentences being empty (if no words match to our word embeddings)
    vectorized_seqs = [x if len(x) > 0 else [0] for x in vectorized_seqs]
    # print(vectorized_seqs)
    vectorized_seqs1 = [[word2idx[tok] for tok in seq if tok in word2idx] for seq in edited_corpus]
    # To avoid any sentences being empty (if no words match to our word embeddings)
    vectorized_seqs1 = [x if len(x) > 0 else [0] for x in vectorized_seqs1]
    meangrade = [x[2] for x in combined_list]
    vectorized_seqs2 = [[word2idx[tok] for tok in seq if tok in word2idx] for seq in combined_list]
    # To avoid any sentences being empty (if no words match to our word embeddings)
    vectorized_seqs2 = [x if len(x) > 0 else [0] for x in vectorized_seqs2]
    vectorized_seqs3 = [[word2idx[tok] for tok in seq if tok in word2idx] for seq in corpus_test]
    # To avoid any sentences being empty (if no words match to our word embeddings)
    vectorized_seqs3 = [x if len(x) > 0 else [0] for x in vectorized_seqs3]
    # print(meangrade)
    # print(vectorized_seqs1)
    INPUT_DIM = len(word2idx)
    BATCH_SIZE = 32
    HIDDEN_DIM = 50

    for x in vectorized_seqs:
        real.append(x)
    for x in vectorized_seqs1:
        real.append(x)
    # print(real)
    meangrade = [x[2] for x in combined_list]
    #model = models.vgg(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, WINDOW_SIZE, OUTPUT_DIM, DROPOUT)
    # model = CNN(INPUT_DIM, EMBEDDING_DIM, N_OUT_CHANNELS, WINDOW_SIZE, OUTPUT_DIM, DROPOUT)
    model=models.TransformerModel(INPUT_DIM, hyperparameters.EMBEDDING_DIM, hyperparameters.SIZES,
                                  hyperparameters.D_HID , hyperparameters.OUTPUT_DIM, hyperparameters.DROPOUT)
    print("Model initialised.")
    criteon = nn.CrossEntropyLoss()
    model.to(device)
    model.embedding.weight.data.copy_(torch.from_numpy(wvecs))
    feature = vectorized_seqs1
    train_and_dev = Task1Dataset(feature, meangrade)
    test_dataset=Task1Dataset(vectorized_seqs3, test_df['meanGrade'])
    train_examples = round(len(train_and_dev) * train_proportion)
    dev_examples = len(train_and_dev) - train_examples

    train_dataset, dev_dataset = random_split(train_and_dev,
                                              (train_examples,
                                               dev_examples))
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE,
                                               collate_fn=collate_fn_padd)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_padd)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE,
                                               collate_fn=collate_fn_padd)
    print("Dataloaders created.")
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    functionDic[model.model_type][0](train_loader, dev_loader, model, epochs,loss_fn,optimizer)
    functionDic[model.model_type][1](test_loader,model,loss_fn)