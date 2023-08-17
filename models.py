import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from transformers import BertModel,BertTokenizer


class CNNMultiLayer(nn.Module):
    # TO DO
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len]

        embedded = self.embedding(x)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        preds = self.fc(cat)
        return preds

class CNN(nn.Module):
    # TO DO
    def __init__(self, vocab_size, embedding_dim, out_channels, window_size, output_dim, dropout):
        super(CNN, self).__init__()

        # Create the embedding layer as usual
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # in_channels -- 1 text channel
        # out_channels -- the number of output channels
        # kernel_size is (window size x embedding dim)
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=out_channels,
            kernel_size=(window_size, embedding_dim))

        # the dropout layer
        self.dropout = nn.Dropout(dropout)

        # the output layer
        self.fc = nn.Linear(out_channels, output_dim)

    def forward(self, x):
        # x -> (batch size, max_sent_length)

        embedded = self.embedding(x)
        # embedded -> (batch size, max_sent_length, embedding_dim)

        # images have 3 RGB channels
        # for the text we add 1 channel
        embedded = embedded.unsqueeze(1)
        # embedded -> (batch size, 1, max_sent_length, embedding dim)

        # Compute the feature maps
        feature_maps = self.conv(embedded)

        ##########################################
        # Q: What is the shape of `feature_maps` ?
        ##########################################
        # A: (batch size, n filters, max_sent_length - window size + 1, 1)

        feature_maps = feature_maps.squeeze(3)

        # Q: why do we remove 1 dimension here?
        # A: we do need the 1 channel anymore

        # Apply ReLU
        feature_maps = F.relu(feature_maps)

        # Apply the max pooling layer
        pooled = F.max_pool1d(feature_maps, feature_maps.shape[2])

        pooled = pooled.squeeze(2)

        ####################################
        # Q: What is the shape of `pooled` ?
        ####################################
        # A: (batch size, n_filters)

        dropped = self.dropout(pooled)
        preds = self.fc(dropped)

        return preds

##new models
class BertForClassification(nn.Module):
    """BERT with simple linear model."""
    def __init__(self, vocab_size,config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.bert_model_path: pretrained model path or model type
                    e.g. 'bert-base-chinese'
                config.hidden_size: The same as BERT model, usually 768
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, config.hidden_size, padding_idx=0)
        self.bert = BertModel.from_pretrained(config.bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.linear = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Forward inputs and get logits.

        Args:
            input_ids: (batch_size, max_seq_len)
            attention_mask: (batch_size, max_seq_len)
            token_type_ids: (batch_size, max_seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.shape[0]
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=False)
        # bert_output[0]: (batch_size, sequence_length, hidden_size)
        # bert_output[1]: (batch_size, hidden_size)
        pooled_output = bert_output[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output).view(batch_size, self.num_classes)
        logits = nn.functional.softmax(logits, dim=-1)
        # logits: (batch_size, num_classes)
        return logits
class RnnForSentencePairClassification(nn.Module):
    """Unidirectional GRU model for sentences pair classification.
    2 sentences use the same encoder and concat to a linear model.
    """
    def __init__(self, config):
        """Initialize the model with config dict.

        Args:
            config: python dict must contains the attributes below:
                config.vocab_size: vocab size
                config.hidden_size: RNN hidden size and embedding dim
                config.num_classes: int, e.g. 2
                config.dropout: float between 0 and 1
        """
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.rnn = nn.GRU(
            config.hidden_size, hidden_size=config.hidden_size,
            bidirectional=False, batch_first=True)
        self.linear = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.num_classes = config.num_classes

    def forward(self, s1_ids, s2_ids, s1_lengths, s2_lengths):
        """Forward inputs and get logits.

        Args:
            s1_ids: (batch_size, max_seq_len)
            s2_ids: (batch_size, max_seq_len)
            s1_lengths: (batch_size)
            s2_lengths: (batch_size)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = s1_ids.shape[0]
        # ids: (batch_size, max_seq_len)
        s1_embed = self.embedding(s1_ids)
        s2_embed = self.embedding(s2_ids)
        # embed: (batch_size, max_seq_len, hidden_size)
        s1_packed: PackedSequence = pack_padded_sequence(
            s1_embed, s1_lengths, batch_first=True, enforce_sorted=False)
        s2_packed: PackedSequence = pack_padded_sequence(
            s2_embed, s2_lengths, batch_first=True, enforce_sorted=False)
        # packed: (sum(lengths), hidden_size)
        self.rnn.flatten_parameters()
        _, s1_hidden = self.rnn(s1_packed)
        _, s2_hidden = self.rnn(s2_packed)
        s1_hidden = s1_hidden.view(batch_size, -1)
        s2_hidden = s2_hidden.view(batch_size, -1)
        hidden = torch.cat([s1_hidden, s2_hidden], dim=-1)
        hidden = self.linear(hidden).view(-1, self.num_classes)
        hidden = self.dropout(hidden)
        logits = nn.functional.softmax(hidden, dim=-1)
        # logits: (batch_size, num_classes)
        return logits
class NeuralNetwork(nn.Module):
    def __init__(self ,vocab_size, embedding_dim, out_channels, window_size, output_dim, dropout):
        super(NeuralNetwork, self).__init__()
        self.model_type='NeuralNetwork'
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.linear_relu_stack = nn.Sequential(

            nn.Linear(embedding_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, output_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        x=self.embedding(x)
        logits = self.linear_relu_stack(x)
        return logits
class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'TransformerModel'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead[0], d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nhead[1])
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model,  nlayers)
        self.decoderconv = nn.Conv2d(
            in_channels=1, out_channels=d_hid,
            kernel_size=(nhead[2], d_model))
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output=output.unsqueeze(1)
        output=self.decoderconv(output)
        output=output.squeeze(3)
        output=F.relu(output)
        output=F.max_pool1d(output,output.shape[2])
        output=output.squeeze(2)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class vgg(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N_FILTERS, window_size, output_dim, dropout):
        super(vgg, self).__init__()
        self.model_type='vgg'
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.net = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=N_FILTERS,
                                            kernel_size=(window_size[0], embedding_dim)), nn.ReLU(True)])
        for i in range(window_size[1] - 1):
            self.net.extend([nn.Conv2d(in_channels=N_FILTERS, out_channels=N_FILTERS,
                                       kernel_size=(window_size[0], 1)), nn.ReLU(True)])
        self.net = nn.Sequential(*self.net)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(N_FILTERS, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        # print(x.shape)
        feature_maps = embedded.unsqueeze(1)
        feature_maps = self.net(feature_maps)
        feature_maps = feature_maps.squeeze(3)
        feature_maps = F.max_pool1d(feature_maps, feature_maps.shape[2])
        feature_maps = feature_maps.squeeze(2)
        dropped = self.dropout(feature_maps)
        preds = self.fc(dropped)

        return preds