import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class BaseRNN(nn.Module):

    # SYM_MASK = "MASK"
    # SYM_EOS = "EOS"

    def __init__(
            self, input_size, hidden_size, input_dropout_p, dropout_p, n_layers,
            rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.dropout = nn.Dropout(p=input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {}".format(rnn_cell))

        self.dropout_p = dropout_p

        self.compress_hn_layers = nn.Linear(
            n_layers * hidden_size, hidden_size, bias=False)
        self.compress_hn_layers_bi = nn.Linear(
            n_layers * 2 * hidden_size, hidden_size, bias=False)
        self.compress_hn_bi = nn.Linear(
            2 * hidden_size, hidden_size, bias=False)

        self.compress_output = nn.Linear(
            2 * hidden_size, hidden_size, bias=False)
        self.compress_output_dropout = nn.Dropout(p=input_dropout_p)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class SentenceEncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    variable_lengths: if use variable length RNN (default: False)

    Args:
        input_var (batch, seq_len): tensor containing the features of the input sequence.
        input_lengths (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch


    Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
    """

    def __init__(
            self,
            vocab_size,
            hidden_size,
            input_dropout_p=0,
            dropout_p=0,
            n_layers=1,
            bidirectional=False,
            rnn_cell='gru',
            variable_lengths=True,
            embedding=None,
            update_embedding=True):
        super().__init__(
            vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers,
            rnn_cell)

        self.variable_lengths = variable_lengths
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        embedding_dim = embedding.shape[
            1] if embedding is not None else hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding is not None:
            # self.embedding.weight.data.copy_(torch.from_numpy(embedding))
            self.embedding.weight = nn.Parameter(
                torch.from_numpy(embedding).float())
        self.upcompress_embedding = nn.Linear(
            embedding_dim, hidden_size, bias=False)

        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_p)

    def forward(self, input_var, h_0=None, input_lengths=None):
        batch_size = input_var.size()[0]
        # nn.Embedding adds a new dim. embedded is of shape (batch_size, seq_len, hidden_size)
        embedded = self.embedding(input_var)
        embedded = self.dropout(embedded)
        embedded = F.relu(self.upcompress_embedding(embedded))

        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths, batch_first=True, enforce_sorted=False)

        # output of shape (batch, seq_len, num_directions * hidden_size)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        output, hidden = self.rnn(embedded, h_0)

        if self.variable_lengths:
            total_length = input_var.size()[1]
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=None)

        if self.n_layers > 1 and self.bidirectional:
            output = self.dropout(F.relu(self.compress_output(output)))
            hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
            hidden = self.dropout(F.relu(self.compress_hn_layers_bi(hidden)))
        elif self.n_layers > 1:
            hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
            hidden = self.dropout(F.relu(self.compress_hn_layers(hidden)))
        elif self.bidirectional:
            output = self.dropout(F.relu(self.compress_output(output)))
            hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
            hidden = self.dropout(F.relu(self.compress_hn_bi(hidden)))
        # output of shape (batch, seq_len, hidden_size) hidden of shape (batch, hidden_size)

        return output, hidden


class VideoEncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input video.

    variable_lengths: if use variable length RNN (default: False)

    Args:
        input_var (batch, seq_len, feature_size): tensor containing the features of the input sequence.
        input_lengths (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch


    Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            input_dropout_p=0,
            dropout_p=0,
            n_layers=1,
            bidirectional=False,
            rnn_cell='gru',
            variable_lengths=True):
        super().__init__(
            None, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.project = nn.Linear(input_size, hidden_size, bias=False)

        self.variable_lengths = variable_lengths
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.rnn = self.rnn_cell(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_p)

    def forward(self, input_var, h_0=None, input_lengths=None):
        batch_size = input_var.size()[0]

        embedded = self.project(input_var)
        embedded = self.dropout(F.relu(embedded))
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths, batch_first=True, enforce_sorted=False)

        output, hidden = self.rnn(embedded, h_0)

        if self.variable_lengths:
            total_length = input_var.size()[1]
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=None)

        if self.n_layers > 1 and self.bidirectional:
            output = self.dropout(F.relu(self.compress_output(output)))
            hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
            hidden = self.dropout(F.relu(self.compress_hn_layers_bi(hidden)))
        elif self.n_layers > 1:
            hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
            hidden = self.dropout(F.relu(self.compress_hn_layers(hidden)))
        elif self.bidirectional:
            output = self.dropout(F.relu(self.compress_output(output)))
            hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)
            hidden = self.dropout(F.relu(self.compress_hn_bi(hidden)))
        # output of shape (batch, seq_len, hidden_size) hidden of shape (batch, hidden_size)

        return output, hidden