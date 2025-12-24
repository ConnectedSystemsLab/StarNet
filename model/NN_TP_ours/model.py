import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
def count_parameters(model):
    total_params = 0
    print('------------------\n Model Parameters\n------------------')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'Layer: {name} | Number of parameters: {param.numel()}')
            total_params += param.numel()
    print(f'Total number of parameters: {total_params}')


class FeatureEmbedding(nn.Module):
    def __init__(self, c_in, c_out):
        super(FeatureEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.Conv = nn.Conv1d(in_channels=c_in, out_channels=c_out,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.Conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


def periodical_embedding(x_seq_attr):
    """
    :param x_seq_attr: Batch size x seq_len x feature_size (12)
    :return:
    """
    with torch.no_grad():
        periods = x_seq_attr[:, :, -4]*0
        indices = [i for i in range(x_seq_attr.shape[2]) if i != x_seq_attr.shape[2]-4]
        x_seq_attr_new = x_seq_attr[:, :, indices]
    return x_seq_attr_new, periods


def smape_loss(y_true, y_pred):
    numerator = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    return torch.mean(numerator / (denominator + 1e-8))


class SatelliteGRU_seq2seq(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, output_len, feature_size, device, positional_scale=200, teacher_forcing_ratio=0.3, n_fea_expanded=4, dropout=0.5):
        '''

        :param seq_len:
        :param hidden_size:
        :param num_layers:
        :param output_len:
        :param feature_size:  attribute size
        :param device:
        :param teacher_forcing_ratio:
        '''
        super(SatelliteGRU_seq2seq, self).__init__()
        self.n_fea_expanded = n_fea_expanded  # 4
        self.device = device
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_len = output_len
        self.feature_size = feature_size
        self.output_size = 1  # Assuming output size is the same as feature size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.fc_attr_encoder = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        # Encoder
        self.gru_encoder = nn.GRU(self.feature_size+1, hidden_size, num_layers, batch_first=True)


        # Pretrain Decoder
        self.pretrain_gru_decoder = nn.GRU(self.feature_size+1, hidden_size, num_layers, batch_first=True)
        self.pretrain_fc_decoder = nn.Sequential(
            nn.Linear(hidden_size, self.feature_size+1)
        )

        # Decoder
        self.gru_decoder = nn.GRU(1, hidden_size, num_layers, batch_first=True)
        self.fc_decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.output_size),
        )

    def freeze_encoder(self):
        for param in self.fc_attr_encoder.parameters():
            param.requires_grad = False
        for param in self.gru_encoder.parameters():
            param.requires_grad = False

    def encode(self, x_input, batch_size):
        h0_encoder = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        encoder_outputs, hidden = self.gru_encoder(x_input, h0_encoder)
        return hidden

    def pretrain_decode(self, x_input, batch_size, hidden):
        outputs = torch.zeros(batch_size, self.seq_len, self.feature_size+1).to(self.device)
        decoder_input = x_input[:, -1].reshape([batch_size, 1, -1])  # Start with the last element of the input sequence
        # Decoder forward pass
        for t in range(self.seq_len-1, -1, -1):
            output, hidden = self.pretrain_gru_decoder(decoder_input, hidden)
            output = self.pretrain_fc_decoder(output.squeeze(1))
            outputs[:, t, :] = output
            decoder_input = output.unsqueeze(1)
            decoder_input = decoder_input.reshape([batch_size, 1, -1])
        return outputs

    def decode(self, x_input, batch_size, hidden):
        outputs = torch.zeros(batch_size, self.output_len, self.output_size).to(self.device)
        decoder_input = x_input[:, -1, 0].reshape([batch_size, 1, 1])  # Start with the last element of the input sequence
        # Decoder forward pass
        for t in range(self.output_len):
            output, hidden = self.gru_decoder(decoder_input, hidden)
            output = self.fc_decoder(output.squeeze(1))
            outputs[:, t, :] = output
            decoder_input = output.unsqueeze(1)
            decoder_input = decoder_input.reshape([batch_size, 1, 1])
        return outputs.squeeze(-1)

    def pretrain_forward(self, x_seq, x_seq_attr, y_seq=None):
        '''
        :param x_seq: (batch_size, seq_len)
        :param x_seq_attr: (batch_size, seq_len, feature_size)
        :param y_seq: (batch_size, output_len, feature_size) (optional)
        :param teacher_forcing_ratio: Probability to use the actual output as the next input
        :return: (batch_size, output_len, output_size)
        '''
        batch_size, trace_len = x_seq.size()
        x_input = torch.cat((x_seq.unsqueeze(-1), x_seq_attr), dim=-1)
        # means = x_input.mean(-1, keepdim=True).detach()
        # x_input = x_input - means
        # stdev = torch.sqrt(
        #     torch.var(x_input, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        # x_input /= stdev

        hidden = self.encode(x_input, batch_size)
        outputs = self.pretrain_decode(x_input, batch_size, hidden)
        return x_input, outputs


    def forward(self, x_seq, x_seq_attr, y_seq=None):
        '''
        :param x_seq: (batch_size, seq_len)
        :param x_seq_attr: (batch_size, seq_len, feature_size)
        :param y_seq: (batch_size, output_len, feature_size) (optional)
        :param teacher_forcing_ratio: Probability to use the actual output as the next input
        :return: (batch_size, output_len, output_size)
        '''
        batch_size, trace_len = x_seq.size()
        x_input = torch.cat((x_seq.unsqueeze(-1), x_seq_attr), dim=-1)
        # means = x_input.mean(-1, keepdim=True).detach()
        # x_input = x_input - means
        # stdev = torch.sqrt(
        #     torch.var(x_input, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        # x_input /= stdev

        hidden = self.encode(x_input, batch_size)
        outputs = self.decode(x_input, batch_size, hidden)
        return outputs, 1

    def encode_forward(self, x_seq, x_seq_attr, y_seq=None):
        '''
        :param x_seq: (batch_size, seq_len)
        :param x_seq_attr: (batch_size, seq_len, feature_size)
        :param y_seq: (batch_size, output_len, feature_size) (optional)
        :param teacher_forcing_ratio: Probability to use the actual output as the next input
        :return: (batch_size, output_len, output_size)
        '''
        batch_size, trace_len = x_seq.size()
        x_input = torch.cat((x_seq.unsqueeze(-1), x_seq_attr), dim=-1)
        # means = x_input.mean(-1, keepdim=True).detach()
        # x_input = x_input - means
        # stdev = torch.sqrt(
        #     torch.var(x_input, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        # x_input /= stdev

        hidden = self.encode(x_input, batch_size)
        outputs = self.decode(x_input, batch_size, hidden)
        return outputs, hidden


class Satellite_FNN(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, output_len, feature_size, device, positional_scale=200, teacher_forcing_ratio=0.3, n_fea_expanded=4, dropout=0.5):
        '''

        :param seq_len:
        :param hidden_size:
        :param num_layers:
        :param output_len:
        :param feature_size:  attribute size
        :param device:
        :param teacher_forcing_ratio:
        '''
        super(Satellite_FNN, self).__init__()
        self.n_fea_expanded = n_fea_expanded  # 4
        self.device = device
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_len = output_len
        self.feature_size = feature_size
        self.output_size = 1  # Assuming output size is the same as feature size
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.fc_feature_compress = nn.Sequential(
            nn.Linear(self.feature_size+1, 1),
            nn.ReLU()
        )
        self.fc_predict = nn.Sequential(
            nn.Linear(self.seq_len, self.output_len)
        )

        # Encoder
        self.gru_encoder = nn.GRU(self.feature_size+1, hidden_size, num_layers, batch_first=True)





    def forward(self, x_seq, x_seq_attr, y_seq=None):
        '''
        :param x_seq: (batch_size, seq_len)
        :param x_seq_attr: (batch_size, seq_len, feature_size)
        :param y_seq: (batch_size, output_len, feature_size) (optional)
        :param teacher_forcing_ratio: Probability to use the actual output as the next input
        :return: (batch_size, output_len, output_size)
        '''
        batch_size, trace_len = x_seq.size()
        x_input = torch.cat((x_seq.unsqueeze(-1), x_seq_attr), dim=-1)
        # means = x_input.mean(-1, keepdim=True).detach()
        # x_input = x_input - means
        # stdev = torch.sqrt(
        #     torch.var(x_input, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        # x_input /= stdev

        x = self.fc_feature_compress(x_input).squeeze()
        outputs = self.fc_predict(x)
        return outputs


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query[:, -1, :].unsqueeze(1)) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, keys):
        # query: [batch_size, 1, hidden_size]
        # keys: [batch_size, seq_len, hidden_size]

        # Apply the linear transformation to keys
        keys_transformed = self.Wa(keys)

        # Compute the scores by dot product
        scores = torch.bmm(query, keys_transformed.transpose(1, 2))

        # Compute the weights
        weights = F.softmax(scores, dim=-1)

        # Compute the context vector
        context = torch.bmm(weights, keys)

        return context, weights


class SatelliteGRU_attention(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, output_len, feature_size, device,
                 teacher_forcing_ratio=0.3, n_fea_expanded=4):
        super(SatelliteGRU_attention, self).__init__()
        self.n_fea_expanded = n_fea_expanded
        self.device = device
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_len = output_len
        self.feature_size = feature_size
        self.output_size = 1  # Assuming output size is the same as feature size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.attention = BahdanauAttention(hidden_size)
        # self.attention = LuongAttention(hidden_size)

        # Encoder
        self.data_embedding = FeatureEmbedding(c_in=feature_size, c_out=self.n_fea_expanded)
        self.gru_encoder = nn.GRU(self.n_fea_expanded + 1, hidden_size, self.num_layers, batch_first=True)

        # Attention
        self.attn = nn.Linear(hidden_size + 1, seq_len)
        self.attn_combine = nn.Linear(hidden_size + self.n_fea_expanded + 1, hidden_size)

        # Decoder
        self.gru_decoder = nn.GRU(self.hidden_size + self.n_fea_expanded + 1, hidden_size, self.num_layers, batch_first=True)
        self.fc_decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.n_fea_expanded + 1),
        )
        self.out = nn.Linear(self.n_fea_expanded + 1, 1)


    def encode(self, x_input, batch_size):
        h0_encoder = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        encoder_outputs, hidden = self.gru_encoder(x_input, h0_encoder)
        return encoder_outputs, hidden

    def decode(self, x_input, encoder_outputs, batch_size, hidden):
        outputs = torch.zeros(batch_size, self.output_len, self.n_fea_expanded+1).to(self.device)
        decoder_input = x_input[:, -1, :].unsqueeze(1)# Start with the last element of the input sequence
        decoder_hidden = hidden
        attentions = []

        for t in range(self.output_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            outputs[:, t, :] = decoder_output.squeeze(1)
            decoder_input = decoder_output.detach()
            attentions.append(attn_weights)

        outputs = self.out(outputs)
        attentions = torch.cat(attentions, dim=1)
        return outputs.squeeze(-1), attentions

    def forward(self, x_seq, x_seq_attr, y_seq=None):
        """
        :param x_seq: Batch size x seq_len
        :param x_seq_attr: Batch size x seq_len x feature_size (12 if full)
        :param y_seq: Batch size x output_len
        :return:
        """
        batch_size, trace_len = x_seq.size()
        # x_seq_attr, period = periodical_embedding(x_seq_attr)

        x_input = torch.cat((x_seq.unsqueeze(-1), x_seq_attr), dim=-1)
        # x_input = self.data_embedding(x_input)
        # x_input = torch.cat((x_input, period.unsqueeze(-1)), dim=-1)

        encoder_outputs, hidden = self.encode(x_input, batch_size)
        outputs, attentions = self.decode(x_input, encoder_outputs, batch_size, hidden)
        return outputs, attentions

    def forward_step(self, decoder_input, hidden, encoder_outputs):
        embedded = decoder_input

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru_decoder(input_gru, hidden)
        output = self.fc_decoder(output)

        return output, hidden, attn_weights

    def encoder_forward(self, x_seq, x_seq_attr, y_seq=None):
        batch_size, trace_len = x_seq.size()
        x_input = torch.cat((x_seq.unsqueeze(-1), x_seq_attr), dim=-1)

        encoder_outputs, hidden = self.encode(x_input, batch_size)
        outputs, attentions = self.decode(x_input, encoder_outputs, batch_size, hidden)
        return outputs, hidden
