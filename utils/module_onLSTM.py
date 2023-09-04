"""
@Author		:           Lee, Qin
@StartTime	:           2018/08/13
@Filename	:           module.py
@Software	:           Pycharm
@Framework  :           Pytorch
@LastModify	:           2019/05/07
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.autograd import Variable

class ModelManager(nn.Module):

    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        self.__num_word = num_word               # 869
        self.__num_slot = num_slot               # 120
        self.__num_intent = num_intent           # 21
        self.__args = args

        # Initialize an embedding object.
        self.__embedding = EmbeddingCollection(
            self.__num_word,                     # 869
            self.__args.word_embedding_dim       # 256
        )

        # Initialize an LSTM Encoder object.
        self.__encoder = LSTMEncoder(
            self.__args.word_embedding_dim,      # 256
            self.__args.encoder_hidden_dim,      # 256
            self.__args.dropout_rate
        )

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.__args.word_embedding_dim,      # 256
            self.__args.attention_hidden_dim,    # 1024
            self.__args.attention_output_dim,    # 128
            self.__args.dropout_rate
        )

        # Initialize an Decoder object for intent.
        self.__intent_decoder = LSTMDecoder(           # 384 to 64
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,  # 256+128
            self.__args.intent_decoder_hidden_dim,                              # 64
            self.__num_intent, self.__args.dropout_rate,                        # 21
            embedding_dim=self.__args.intent_embedding_dim                      # 8
        )
        # Initialize an Decoder object for slot.
        self.__slot_decoder = LSTMDecoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,  # 256+128
            self.__args.slot_decoder_hidden_dim,                                # 64
            self.__num_slot, self.__args.dropout_rate,                          # 120
            embedding_dim=self.__args.slot_embedding_dim,                       # 32
            extra_dim=self.__num_intent                                         # 21
        )

        # One-hot encoding for augment data feed. 
        self.__intent_embedding = nn.Embedding(  # 64 to 64 ?
            self.__num_intent, self.__num_intent
        )
        self.__intent_embedding.weight.data = torch.eye(self.__num_intent)      # 对角线全1矩阵
        self.__intent_embedding.weight.requires_grad = False
###
        self.wa = nn.Parameter(torch.zeros(size=(384,self.__num_intent)))
        nn.init.xavier_uniform_(self.wa.data, gain=1.414)  # xavier初始化
###
    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))
        print('\tword embedding dimension:				    {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				    {};'.format(self.__args.encoder_hidden_dim))
        print('\tdimension of intent embedding:		    	{};'.format(self.__args.intent_embedding_dim))
        print('\tdimension of slot embedding:			    {};'.format(self.__args.slot_embedding_dim))
        print('\tdimension of slot decoder hidden:  	    {};'.format(self.__args.slot_decoder_hidden_dim))
        print('\tdimension of intent decoder hidden:        {};'.format(self.__args.intent_decoder_hidden_dim))
        print('\thidden dimension of self-attention:        {};'.format(self.__args.attention_hidden_dim))
        print('\toutput dimension of self-attention:        {};'.format(self.__args.attention_output_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def forward(self, text, seq_lens, n_predicts=None, forced_slot=None, forced_intent=None):
        """
        seq_lens: [18, 16, 15, 12, 10, 10, 10, 9, 9, 9, 9, 8, 8, 7, 7, 7] 和为164
        """
        word_tensor, _ = self.__embedding(text)     # torch.Size([16, 18, 256])
        lstm_hiddens = self.__encoder(word_tensor, seq_lens)    # torch.Size([164, 256])
        # transformer_hiddens = self.__transformer(pos_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)     # torch.Size([164, 128])
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=1)           # 公式2, torch.Size([164, 384]) e
        # print(1 / 0)
        pred_intent = self.__intent_decoder(     # torch.Size([164, 21])
            hiddens, seq_lens,
            forced_input=forced_intent      # None?
        )
        ###
        h=[]
        t = []
        for i in range(0,hiddens.shape[0]):
            h = torch.unsqueeze(hiddens, 1)     # torch.Size([seq, 1, 384])
            # print(self.wa.shape)

            # print(torch.tensor(hiddens[i]).shape) 384
            # print(torch.tensor(self.__Wa).shape)
            # h.append(hiddens[i].unsqueeze(1))
            # h=h[i]
            # print(h)
            # print(1/0)
            # h[i] = torch.tensor(hiddens[i]).unsqueeze(1)
            # pred_intent[i] = torch.t(pred_intent.squeeze(1))
            # print(hiddens[i].shape)
            # t.append(torch.mm(torch.mm(hiddens[i], self.wa), pred_intent[i]))
            # print(torch.tensor(t).shape)
        # print(1/0)
        ###
        if not self.__args.differentiable:
            _, idx_intent = pred_intent.topk(1, dim=-1)     # torch.Size([164, 1])
            #print(idx_intent)
            feed_intent = self.__intent_embedding(idx_intent.squeeze(1))    # torch.Size([164, 21])
            # print("m")
            # print(feed_intent.shape)
            # print(1 / 0)
        else:
            feed_intent = pred_intent

        pred_slot = self.__slot_decoder(
            hiddens, seq_lens,
            forced_input=forced_slot,
            extra_input=feed_intent
        )

        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_intent, dim=1)
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            _, intent_index = pred_intent.topk(n_predicts, dim=1)

            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()

    # def golden_intent_predict_slot(self, text, seq_lens, golden_intent, n_predicts=1):
    #     word_tensor, _ = self.__embedding(text)
    #     embed_intent = self.__intent_embedding(golden_intent)
    #
    #     lstm_hiddens = self.__encoder(word_tensor, seq_lens)
    #     attention_hiddens = self.__attention(word_tensor, seq_lens)
    #     hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=1)
    #
    #     pred_slot = self.__slot_decoder(
    #         hiddens, seq_lens, extra_input=embed_intent
    #     )
    #     _, slot_index = pred_slot.topk(n_predicts, dim=-1)
    #
    #     # Just predict single slot value.
    #     return slot_index.cpu().data.numpy().tolist()


class EmbeddingCollection(nn.Module):
    """
    Provide word vector and position vector encoding.
    """

    def __init__(self, input_dim, embedding_dim, max_len=5000):  # ATIS：869 to 256
        super(EmbeddingCollection, self).__init__()

        self.__input_dim = input_dim
        # Here embedding_dim must be an even embedding.
        self.__embedding_dim = embedding_dim
        self.__max_len = max_len

        # Word vector encoder.
        self.__embedding_layer = nn.Embedding(
            self.__input_dim, self.__embedding_dim
        )

        # Position vector encoder.
        # self.__position_layer = torch.zeros(self.__max_len, self.__embedding_dim)
        # position = torch.arange(0, self.__max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, self.__embedding_dim, 2) *
        #                      (-math.log(10000.0) / self.__embedding_dim))

        # Sine wave curve design.
        # self.__position_layer[:, 0::2] = torch.sin(position * div_term)
        # self.__position_layer[:, 1::2] = torch.cos(position * div_term)
        #
        # self.__position_layer = self.__position_layer.unsqueeze(0)
        # self.register_buffer('pe', self.__position_layer)

    def forward(self, input_x):
        # Get word vector encoding.
        embedding_x = self.__embedding_layer(input_x)

        # Get position encoding.
        # position_x = Variable(self.pe[:, :input_x.size(1)], requires_grad=False)

        # Board-casting principle.
        return embedding_x, embedding_x


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):  # 256  256
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim        # 256
        self.__hidden_dim = hidden_dim // 2         # 128
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        # self.__lstm_layer = nn.LSTM(
        #     input_size=self.__embedding_dim,        # 256
        #     hidden_size=self.__hidden_dim,          # 128
        #     batch_first=True,
        #     bidirectional=True,
        #     dropout=self.__dropout_rate,
        #     num_layers=1
        # )

        self.__lstm_layer = ONLSTMStack(
            input_size=self.__embedding_dim,        # 256
            hidden_size=self.__hidden_dim,          # 128
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.填充和嵌入的输入文本。
        :param seq_lens: is the length of original input text.是原始输入文本的长度。
        :return: is encoded word hidden vectors. 是编码的词隐藏向量。
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # # Pack and Pad process for input of variable length. 可变长度输入的打包和填充过程。
        # packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        # lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        # padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)
        #
        # return torch.cat([padded_hiddens[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)

        lstm_hiddens, (_) = self.__lstm_layer(dropout_text)

        return torch.cat([lstm_hiddens[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)


class LSTMDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, embedding_dim=None, extra_dim=None):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.    解码器的输入维度。实际上，它是编码器隐藏大小。
        :param hidden_dim: hidden dimension of iterative LSTM.                               迭代 LSTM 的隐藏维度。
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.   解码器的输出维度。实际上，它是意图或槽的总数。
        :param dropout_rate: dropout rate of network which is only useful for embedding.        仅对嵌入网络有用的丢失率。
        :param embedding_dim: if it's not None, the input and output are relevant.              如果不是None，则输入和输出是相关的。
        :param extra_dim: if it's not None, the decoder receives information tensors.           如果不是None，解码器接收信息张量。
        """

        super(LSTMDecoder, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim  # 64
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(
                torch.randn(1, self.__embedding_dim),
                requires_grad=True
            )

        # Make sure the input dimension of iterative LSTM.  确保迭代 LSTM 的输入维度。
        if self.__extra_dim is not None and self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim + self.__embedding_dim
        elif self.__extra_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim
        elif self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__embedding_dim
        else:
            lstm_input_dim = self.__input_dim

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)

        # self.__lstm_layer = nn.LSTM(
        #     input_size=lstm_input_dim,
        #     hidden_size=self.__hidden_dim,
        #     batch_first=True,
        #     bidirectional=False,
        #     dropout=self.__dropout_rate,
        #     num_layers=1
        # )

        self.__lstm_layer = ONLSTMStack(
            input_size=lstm_input_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__dropout_rate,
            num_layers=1
        )

        self.__linear_layer = nn.Linear(
            self.__hidden_dim,
            self.__output_dim
        )

    def forward(self, encoded_hiddens, seq_lens, forced_input=None, extra_input=None):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder. 是编码器产生的编码隐藏张量。
        :param seq_lens: is a list containing lengths of sentence.               是一个包含句子长度的列表。
        :param forced_input: is truth values of label, provided by teacher forcing.     是标签的真值，由教师强制提供。
        :param extra_input: comes from another decoder as information tensor.       来自另一个解码器作为信息张量。
        :return: is distribution of prediction labels.                  是预测标签的分布。
        """

        # Concatenate information tensor if possible.
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=1)
        else:
            input_tensor = encoded_hiddens

        output_tensor_list, sent_start_pos = [], 0
        if self.__embedding_dim is None or forced_input is not None:

            for sent_i in range(0, len(seq_lens)):
                sent_end_pos = sent_start_pos + seq_lens[sent_i]

                # Segment input hidden tensors.
                seg_hiddens = input_tensor[sent_start_pos: sent_end_pos, :]

                if self.__embedding_dim is not None and forced_input is not None:
                    if seq_lens[sent_i] > 1:
                        seg_forced_input = forced_input[sent_start_pos: sent_end_pos]
                        seg_forced_tensor = self.__embedding_layer(seg_forced_input).view(seq_lens[sent_i], -1)
                        seg_prev_tensor = torch.cat([self.__init_tensor, seg_forced_tensor[:-1, :]], dim=0)
                    else:
                        seg_prev_tensor = self.__init_tensor

                    # Concatenate forced target tensor.
                    combined_input = torch.cat([seg_hiddens, seg_prev_tensor], dim=1)
                else:
                    combined_input = seg_hiddens
                dropout_input = self.__dropout_layer(combined_input)

                lstm_out, _ = self.__lstm_layer(dropout_input.view(1, seq_lens[sent_i], -1))
                lstm_out, _ = self.__lstm_layer(dropout_input.view(1, seq_lens[sent_i], -1))
                linear_out = self.__linear_layer(lstm_out.view(seq_lens[sent_i], -1))

                output_tensor_list.append(linear_out)
                sent_start_pos = sent_end_pos
        else:
            for sent_i in range(0, len(seq_lens)):
                prev_tensor = self.__init_tensor

                # It's necessary to remember h and c state  需要记住h和c状态
                # when output prediction every single step. 每一步都输出预测。
                last_h, last_c = None, None

                sent_end_pos = sent_start_pos + seq_lens[sent_i]
                for word_i in range(sent_start_pos, sent_end_pos):
                    seg_input = input_tensor[[word_i], :]
                    combined_input = torch.cat([seg_input, prev_tensor], dim=1)
                    dropout_input = self.__dropout_layer(combined_input).view(1, 1, -1)

                    # if last_h is None and last_c is None:
                    #     lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input)
                    # else:
                    #     lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input, (last_h, last_c))

                    if last_h is None and last_c is None:
                        lstm_out, (_) = self.__lstm_layer(dropout_input)
                    else:
                        lstm_out, (_) = self.__lstm_layer(dropout_input, (_))

                    lstm_out = self.__linear_layer(lstm_out.view(1, -1))
                    output_tensor_list.append(lstm_out)

                    _, index = lstm_out.topk(1, dim=1)
                    prev_tensor = self.__embedding_layer(index).view(1, -1)
                sent_start_pos = sent_end_pos

        return torch.cat(output_tensor_list, dim=0)


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    基于 Query-Key-Value 架构的注意力机制。特别是当query == key == value 时，它是self-attention。
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention. 注意力的前向传播。

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ) / math.sqrt(self.__hidden_dim), dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        flat_x = torch.cat(
            [attention_x[i][:seq_lens[i], :] for
             i in range(0, len(seq_lens))], dim=0
        )
        return flat_x


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class LinearDropConnect(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dropout=0.):
        super(LinearDropConnect, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.dropout = dropout

    def sample_mask(self):
        if self.dropout == 0.:
            self._weight = self.weight
        else:
            mask = self.weight.new_empty(
                self.weight.size(),
                dtype=torch.uint8
            )
            mask.bernoulli_(self.dropout)
            self._weight = self.weight.masked_fill(mask, 0.)

    def forward(self, input, sample_mask=False):
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(input, self._weight, self.bias)
        else:
            return F.linear(input, self.weight * (1 - self.dropout),
                            self.bias)


def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)


class ONLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, chunk_size, dropconnect=0.):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)

        # 4 * hidden_size 表示 遗忘、记忆、输出门的计算，以及本层信息提取， 只不过还没有加入激活函数
        # 使用一个线性函数可以简化模型，高速度
        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size + self.n_chunk * 2, bias=True),
            # LayerNorm(3 * hidden_size)
        )
        self.hh = LinearDropConnect(hidden_size, hidden_size * 4 + self.n_chunk * 2, bias=True, dropout=dropconnect)

        # self.c_norm = LayerNorm(hidden_size)

        self.drop_weight_modules = [self.hh]

    def forward(self, input, hidden, transformed_input=None):
        hx, cx = hidden

        if transformed_input is None:
            transformed_input = self.ih(input)

        gates = transformed_input + self.hh(hx)
        # chunk 第一参数表示划分几个chunk，第二参数表示在哪个维度划分
        cingate, cforgetgate = gates[:, :self.n_chunk * 2].chunk(2, 1)
        outgate, cell, ingate, forgetgate = gates[:, self.n_chunk * 2:].view(-1, self.n_chunk * 4,
                                                                             self.chunk_size).chunk(4, 1)

        cingate = 1. - cumsoftmax(cingate)
        cforgetgate = cumsoftmax(cforgetgate)

        distance_cforget = 1. - cforgetgate.sum(dim=-1) / self.n_chunk
        distance_cin = cingate.sum(dim=-1) / self.n_chunk

        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cell = F.tanh(cell)
        outgate = F.sigmoid(outgate)

        # cy = cforgetgate * forgetgate * cx + cingate * ingate * cell

        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell

        # hy = outgate * F.tanh(self.c_norm(cy))
        hy = outgate * F.tanh(cy)
        return hy.view(-1, self.hidden_size), cy, (distance_cforget, distance_cin)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.hidden_size).zero_(),
                weight.new(bsz, self.n_chunk, self.chunk_size).zero_())

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


class ONLSTMStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, chunk_size=32, bidirectional=False, batch_first=False,
                 dropout=0., dropconnect=0.):
        super(ONLSTMStack, self).__init__()
        layer_sizes = [input_size, hidden_size] + [hidden_size for i in range(num_layers - 1)]
        self.cells = nn.ModuleList([ONLSTMCell(layer_sizes[i],
                                               layer_sizes[i + 1],
                                               chunk_size,
                                               dropconnect=dropconnect)
                                    for i in range(len(layer_sizes) - 1)])
        self.lockdrop = LockedDropout()
        self.dropout = dropout
        self.sizes = layer_sizes
        self.bidirectional = bidirectional

    def init_hidden(self, bsz):
        return [c.init_hidden(bsz) for c in self.cells]

    def real_forward(self,input):
        length, batch_size, _ = input.size()
        hidden = self.init_hidden(batch_size)

        if self.training:
            for c in self.cells:
                c.sample_masks()

        prev_state = list(hidden)
        prev_layer = input

        raw_outputs = []
        outputs = []
        distances_forget = []
        distances_in = []
        for l in range(len(self.cells)):
            curr_layer = [None] * length
            dist = [None] * length
            t_input = self.cells[l].ih(prev_layer)

            for t in range(length):
                hidden, cell, d = self.cells[l](
                    None, prev_state[l],
                    transformed_input=t_input[t]
                )
                prev_state[l] = hidden, cell  # overwritten every timestep
                curr_layer[t] = hidden
                dist[t] = d

            prev_layer = torch.stack(curr_layer)
            dist_cforget, dist_cin = zip(*dist)
            dist_layer_cforget = torch.stack(dist_cforget)
            dist_layer_cin = torch.stack(dist_cin)
            raw_outputs.append(prev_layer)
            if l < len(self.cells) - 1:
                prev_layer = self.lockdrop(prev_layer, self.dropout)
            outputs.append(prev_layer)
            distances_forget.append(dist_layer_cforget)
            distances_in.append(dist_layer_cin)
        output = prev_layer
        # [T, B, hidden_size]
        # return output, prev_state, raw_outputs, outputs, (torch.stack(distances_forget), torch.stack(distances_in))
        return output, prev_state

    def forward(self, input):
        """ no pad:
        input:  torch.Size([16, 45, 128])
        """
        """
        len(input)=4
        input[1]: ([16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 14, 12, 11,
        11, 11, 10, 10,  9,  9,  9,  9,  9,  8,  7,  6,  5,  3,  3,  3,  3,  3,
         2,  2,  2,  2,  2,  2,  2,  2,  1]),torch.Size([45])
        input:[[torch.Size([419, 128])]   [ torch.Size([45])]]
        """
        # input [B, T, F]
        # input = input[:, :, :256].pLermute(1, 0, 2)

        r_input = torch.flip(input, dims=[-1])

        output,prev_state = self.real_forward(input)
        if self.bidirectional:
            output2,prev_state2 = self.real_forward(r_input)


            output = torch.cat([output,output2],dim=-1)
            # prev_state = torch.cat([prev_state,prev_state2],dim=-1)
            # print(output.shape)
            # print(1/0)
            return output, prev_state
        else:
            return output, prev_state
        # return output.permute(1, 0, 2), prev_state
