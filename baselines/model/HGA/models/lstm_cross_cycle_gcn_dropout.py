""" model 7: all code for adjust, copy from model 6
fc(c3d, resnet) -> v_biLSTM
embed+fc(question) -> q_biLSTM
block(v_last_hidden, q_last_hidden) -> global_out
gcn(v_output, q_output) -> local_out
crossover_transformer(q_output, v_output) -> crossover_out
fc(global_out, local_out, crossover_out) -> result

cycle_loss(q_output, v_output) -> cycle_loss

Dropout in Linear, LSTM, block, GCN
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from block import fusions
from . import torchnlp_nn as nlpnn

from .rnn_encoder import VideoEncoderRNN, SentenceEncoderRNN
# from .rnn_encoder import VideoEncoderRNN, SentenceEncoderRNN
from .q_v_transformer import CrossoverTransformer, MaskedCrossoverTransformer, SelfTransformer, SelfTransformerEncoder, SelfAttention, CoAttention, SingleAttention, CoConcatAttention, CoSiameseAttention
from .gcn import AdjLearner, GCN, EvoAdjLearner

# torch.set_printoptions(threshold=np.inf)


class LSTMCrossCycleGCNDropout(nn.Module):

    def __init__(
            self,
            vocab_size,
            s_layers,
            s_bidirectional,
            s_rnn_cell,
            s_embedding,
            resnet_input_size,
            c3d_input_size,
            v_layers,
            v_bidirectional,
            v_rnn_cell,
            hidden_size,
            dropout_p=0.0,
            gcn_layers=2,
            num_heads=8,
            answer_vocab_size=None,
            q_max_len=35,
            v_max_len=80,
            tf_layers=2,
            two_loss=False,
            fusion_type='none',
            ablation='none'):
        super().__init__()

        self.model_name = 'TwoLSTMandBlock'
        self.task = 'none'
        self.tf_layers = tf_layers
        self.two_loss = two_loss
        self.fusion_type = fusion_type
        self.ablation = ablation
        v_input_size = resnet_input_size
        self.q_max_len = q_max_len
        self.v_max_len = v_max_len

        self.dropout = nn.Dropout(p=dropout_p)

        self.q_input_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.v_input_ln = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.sentence_encoder = SentenceEncoderRNN(
            vocab_size,
            hidden_size,
            input_dropout_p=dropout_p,
            dropout_p=dropout_p,
            n_layers=s_layers,
            bidirectional=s_bidirectional,
            rnn_cell=s_rnn_cell,
            embedding=s_embedding)

        self.compress_c3d = nlpnn.WeightDropLinear(
            c3d_input_size,
            resnet_input_size,
            weight_dropout=dropout_p,
            bias=False)
        # self.video_fusion = fusions.Block(
        #     [v_input_size, v_input_size], v_input_size)
        self.video_fusion = nlpnn.WeightDropLinear(
            2 * v_input_size,
            v_input_size,
            weight_dropout=dropout_p,
            bias=False)

        self.video_encoder = VideoEncoderRNN(
            v_input_size,
            hidden_size,
            input_dropout_p=dropout_p,
            dropout_p=dropout_p,
            n_layers=v_layers,
            bidirectional=v_bidirectional,
            rnn_cell=v_rnn_cell)

        self.transformer_encoder = SelfTransformerEncoder(
            hidden_size,
            tf_layers,
            dropout_p,
            vocab_size,
            q_max_len,
            v_max_len,
            embedding=s_embedding,
            position=True)

        # ! masked
        self.crossover_transformer = MaskedCrossoverTransformer(
            q_max_len=q_max_len,
            v_max_len=v_max_len,
            num_heads=8,
            num_layers=tf_layers,
            dropout=dropout_p)

        self.q_transformer = SelfTransformer(
            q_max_len,
            num_heads=8,
            num_layers=tf_layers,
            dropout=dropout_p,
            position=False)
        self.v_transformer = SelfTransformer(
            v_max_len,
            num_heads=8,
            num_layers=tf_layers,
            dropout=dropout_p,
            position=False)

        self.q_selfattn = SelfAttention(
            hidden_size, n_layers=tf_layers, dropout_p=dropout_p)
        self.v_selfattn = SelfAttention(
            hidden_size, n_layers=tf_layers, dropout_p=dropout_p)

        self.co_attn = CoAttention(
            hidden_size, n_layers=tf_layers, dropout_p=dropout_p)

        self.single_attn_semantic = SingleAttention(
            hidden_size, n_layers=tf_layers, dropout_p=dropout_p)

        self.single_attn_visual = SingleAttention(
            hidden_size, n_layers=tf_layers, dropout_p=dropout_p)

        self.co_concat_attn = CoConcatAttention(
            hidden_size, n_layers=tf_layers, dropout_p=dropout_p)

        self.co_siamese_attn = CoSiameseAttention(
            hidden_size, n_layers=tf_layers, dropout_p=dropout_p)

        self.crossover_fusion = fusions.Block(
            [hidden_size, hidden_size], hidden_size, dropout_input=dropout_p)

        self.adj_learner = AdjLearner(
            hidden_size, hidden_size, dropout=dropout_p)
        # self.evo_adj_learner = EvoAdjLearner(
        #     hidden_size, hidden_size, dropout=dropout_p)

        self.gcn = GCN(
            hidden_size,
            hidden_size,
            hidden_size,
            num_layers=gcn_layers,
            dropout=dropout_p)
        self.gcn_atten_pool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=-1))

        self.video_adj_learner = AdjLearner(
            v_input_size, v_input_size, dropout=dropout_p)
        self.video_gcn = GCN(
            v_input_size,
            v_input_size,
            v_input_size,
            num_layers=1,
            dropout=dropout_p)

        self.video_coattn = CoAttention(
            v_input_size, n_layers=1, dropout_p=dropout_p)

        self.global_fusion = fusions.Block(
            [hidden_size, hidden_size], hidden_size, dropout_input=dropout_p)

        if answer_vocab_size is not None:
            self.fusion = fusions.Block(
                [hidden_size, hidden_size], answer_vocab_size)
            self.fc_fusion = nn.Linear(hidden_size, answer_vocab_size)
        else:
            self.fusion = fusions.Block([hidden_size, hidden_size], 1)
            self.fc_fusion = nn.Linear(hidden_size, 1)

    def forward(self, task, *args):
        # expected sentence_inputs is of shape (batch_size, sentence_len, 1)
        # expected video_inputs is of shape (batch_size, frame_num, video_feature)
        self.task = task
        if task == 'Count':
            return self.forward_count(*args)
        elif task == 'FrameQA':
            
            return self.forward_frameqa(*args)
        elif task == 'Action' or task == 'Trans':
            return self.forward_trans_or_action(*args)

    def model_block(
            self, resnet_inputs, c3d_inputs, video_length, sentence_inputs,
            question_length):

        ### question encoder
        q_output, s_hidden = self.sentence_encoder(
            sentence_inputs, input_lengths=question_length)
        # s_last_hidden of shape (batch_size, hidden_size)
        s_last_hidden = torch.squeeze(s_hidden)

        ### video encoder
        c3d_inputs = F.relu(self.compress_c3d(c3d_inputs))

        ## video coattn
        # resnet_inputs, c3d_inputs = self.video_coattn(resnet_inputs, c3d_inputs)

        ## video gcn
        # video_adj = self.video_adj_learner(resnet_inputs, c3d_inputs)
        # r_c_inputs = torch.cat((resnet_inputs, c3d_inputs), dim=1)
        # video_gcn_out = self.video_gcn(r_c_inputs, video_adj)
        # resnet_inputs = video_gcn_out[:, :80, :]
        # c3d_inputs = video_gcn_out[:, 80:, :]

        ## old video encoder(compress 2 times) -> better
        video_inputs = F.relu(
            self.video_fusion(torch.cat((resnet_inputs, c3d_inputs), dim=2)))
        v_output, v_hidden = self.video_encoder(
            video_inputs, input_lengths=video_length)

        ## new video encoder(compress 1 times)-> bad performance
        # v_output, v_hidden = self.video_encoder(
        #     [resnet_inputs, c3d_inputs], input_lengths=video_length)

        v_last_hidden = torch.squeeze(v_hidden)

        ### transformer encoder
        # q_output, v_output = self.transformer_encoder(
        #     sentence_inputs, resnet_inputs, c3d_inputs, question_length,
        #     video_length)
        # s_last_hidden = torch.mean(q_output, dim=1)
        # v_last_hidden = torch.mean(v_output, dim=1)

        if self.ablation != 'local':
            ### question video fusion
            if self.tf_layers != 0:
                q_output = self.q_input_ln(q_output)
                v_output = self.v_input_ln(v_output)

                ### self attention
                if 'self' in self.fusion_type:
                    q_output, _ = self.q_selfattn(q_output)
                    v_output, _ = self.v_selfattn(v_output)

                ### co attention
                if 'coattn' in self.fusion_type:
                    q_output, v_output = self.co_attn(q_output, v_output)

                elif self.fusion_type == 'none':
                    pass

                else:
                    print('unknown attention module')
                    assert 1 == 0

            ### GCN
            adj = self.adj_learner(q_output, v_output)
            ## evo adj learner: two different fc and softmax
            # adj = self.evo_adj_learner(q_output, v_output)

            # q_v_inputs of shape (batch_size, q_v_len, hidden_size)
            q_v_inputs = torch.cat((q_output, v_output), dim=1)
            # q_v_output of shape (batch_size, q_v_len, hidden_size)
            q_v_output = self.gcn(q_v_inputs, adj)

            ### pool GCN
            # # local_out of shape (batch_size, hidden_size)
            # local_out = torch.mean(q_v_output, dim=1)

            ## attention pool
            local_attn = self.gcn_atten_pool(q_v_output)
            local_out = torch.sum(q_v_output * local_attn, dim=1)

        if self.ablation != 'global':
            ### global fusion, (batch, hidden_size)
            global_out = self.global_fusion((s_last_hidden, v_last_hidden))

            if self.ablation != 'local':
                ### output layer. out of shape (batch_size, 1) or (batch_size, num_class)
                out = self.fusion((global_out, local_out))
                # out = self.fc_fusion(
                #     torch.cat((global_out, local_out, crossover_out), dim=1))
            else:
                out = self.fc_fusion(global_out)
        else:
            out = self.fc_fusion(local_out)

        # out of shape (batch_size, ) or (batch_size, num_class)
        out = torch.squeeze(out)

        return out, adj

    def forward_count(
            self, resnet_inputs, c3d_inputs, video_length, sentence_inputs,
            question_length, answers):
        all_adj = torch.zeros(
            resnet_inputs.size(0), self.q_max_len + self.v_max_len,
            self.q_max_len + self.v_max_len)
        # out of shape (batch_size, )
        out, adj = self.model_block(
            resnet_inputs, c3d_inputs, video_length, sentence_inputs,
            question_length)
        all_adj[:, :adj.size(1), :adj.size(2)] = adj
        predictions = torch.clamp(torch.round(out), min=1, max=10).long()
        # answers of shape (batch_size, )
        return out, predictions, answers, all_adj

    def forward_frameqa(
            self, resnet_inputs, c3d_inputs, video_length, sentence_inputs,
            question_length, answers, answers_type):

        all_adj = torch.zeros(
            resnet_inputs.size(0), self.q_max_len + self.v_max_len,
            self.q_max_len + self.v_max_len)
        
        # out of shape (batch_size, num_class)
        out, adj = self.model_block(
            resnet_inputs, c3d_inputs, video_length, sentence_inputs,
            question_length)

        all_adj[:, :adj.size(1), :adj.size(2)] = adj

        _, max_idx = torch.max(out, 1)
        # (batch_size, ), dtype is long
        predictions = max_idx
        # answers of shape (batch_size, )
        return out, predictions, answers, all_adj

    def forward_trans_or_action(
            self, resnet_inputs, c3d_inputs, video_length, candidate_inputs,
            candidate_length, answers, row_index, question_inputs,
            question_length, raw_candidate_inputs, raw_candidate_length):
        candidate_inputs = candidate_inputs.permute(1, 0, 2)
        candidate_length = candidate_length.permute(1, 0)
        all_adj = torch.zeros(
            resnet_inputs.size(0), 5, self.q_max_len + self.v_max_len,
            self.q_max_len + self.v_max_len)
        all_out = []
        for idx, candidate in enumerate(candidate_inputs):
            # out of shape (batch_size, ), adj of shape (batch_size, q_v_len, q_v_len)
            out, adj = self.model_block(
                resnet_inputs, c3d_inputs, video_length, candidate,
                candidate_length[idx])
            all_out.append(out)
            all_adj[:, idx, :adj.size(1), :adj.size(2)] = adj
        # all_out of shape (batch_size, 5)
        all_out = torch.stack(all_out, 0).transpose(1, 0)
        # all_adj of shape (batch, 5, q_v_len, q_v_len)
        _, max_idx = torch.max(all_out, 1)
        # (batch_size, )
        predictions = max_idx

        # answers of shape (batch_size, )
        return all_out, predictions, answers, all_adj