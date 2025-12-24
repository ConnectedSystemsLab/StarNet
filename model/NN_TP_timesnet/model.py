import torch
import torch.nn as nn
import torch.optim as optim
import math
from layers.Conv_Blocks import Inception_Block_V1
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

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, period: int = 15, dropout: float = 0.5, positional_scale=200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.period = period
        self.positional_scale = positional_scale
        position = torch.arange(period).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(period, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        seq_len = x.size(1)
        if seq_len % self.period != 0:
            raise ValueError(f"seq_len ({seq_len}) is not a multiple of period ({self.period})")

        x = x.permute(1, 0, 2)  # Shape: [seq_len, batch_size, embedding_dim]
        pe = self.pe[:self.period]  # Shape: [period, 1, d_model]
        pe = pe.repeat(seq_len // self.period, 1, 1)  # Shape: [seq_len, 1, d_model]
        x = x + pe*self.positional_scale
        x = self.dropout(x)
        return x.permute(1, 0, 2)  # Shape: [batch_size, seq_len, embedding_dim]


class SatEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SatEmbedding, self).__init__()

        minute_size = 60
        hour_size = 24
        weekday_size = 7
        Embed = nn.Embedding
        self.d_model = d_model
        self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.satellite_embed = nn.Linear(7, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        '''

        :param x: [512, 15, 10], in 10, [:4]satellite info, [4:7] weather info, [7:10] time info
        :return:
        '''
        batchsize = x.size(0)
        time_matrix = x[:, :, -3:].long()
        minute_x = self.minute_embed(time_matrix[:, :, -3].unsqueeze(-1).repeat(1, 1, 15).reshape(batchsize, -1))
        hour_x = self.hour_embed(time_matrix[:, :, -2].unsqueeze(-1).repeat(1, 1, 15).reshape(batchsize, -1))
        weekday_x = self.weekday_embed(time_matrix[:, :, -1].unsqueeze(-1).repeat(1, 1, 15).reshape(batchsize, -1))
        time_embedding = self.layer_norm(hour_x + weekday_x + minute_x)

        sat_matrix = x[:, :, :7]

        satellite_embedding = self.satellite_embed(sat_matrix).unsqueeze(-2).repeat(1, 1, 15, 1).reshape(batchsize, -1, self.d_model)
        satellite_embedding = self.layer_norm(satellite_embedding)
        return time_embedding + satellite_embedding

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
        self.fc = nn.Linear(c_out, c_out)
    def forward(self, x):
        x = self.Conv(x.permute(0, 2, 1)).transpose(1, 2)
        x = self.fc(x)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.5):
        super(DataEmbedding, self).__init__()

        self.value_embedding = FeatureEmbedding(c_in=c_in, c_out=c_out)
        self.position_embedding = PositionalEmbedding(d_model=c_out)
        self.temporal_embedding = SatEmbedding(d_model=c_out)# TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        # return self.dropout(x)
        return x


def smape_loss(y_true, y_pred):
    numerator = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
    return torch.mean(numerator / (denominator + 1e-8))

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.top_k)

        res = []
        for i in range(self.top_k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class SatTimesNet(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, output_len, feature_size, device, positional_scale=200, n_fea_expanded=4, dropout=0.5):
        '''

        :param seq_len:
        :param hidden_size:
        :param num_layers:
        :param output_len:
        :param feature_size:  attribute size
        :param device:
        :param teacher_forcing_ratio:
        '''
        super(SatTimesNet, self).__init__()
        self.n_fea_expanded = n_fea_expanded  # 4
        self.device = device
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_len = output_len
        self.feature_size = feature_size
        self.output_size = 1  # Assuming output size is the same as feature size
        self.layer_norm = nn.LayerNorm(self.n_fea_expanded+1)

        self.fc_attr_encoder = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.seq_len),
        )

        padding = 1 if torch.__version__ >= '1.5.0' else 2

        self.Conv = FeatureEmbedding(c_in=1, c_out=self.n_fea_expanded)
        self.positional_encoding = PeriodicPositionalEncoding(d_model=self.n_fea_expanded, period=15, dropout=dropout, positional_scale=positional_scale)

        self.predict_linear = nn.Linear(self.seq_len, output_len + self.seq_len)

        self.timesnet = nn.ModuleList([TimesBlock(seq_len, output_len, 3, self.n_fea_expanded+1, 256, 4)
                                    for _ in range(2)])
        self.fc_projection = nn.Linear(self.n_fea_expanded+1, self.output_size)

    def forward(self, x_seq, x_seq_attr, y_seq=None):
        '''
        :param x_seq: (batch_size, seq_len)
        :param x_seq_attr: (batch_size,feature_size)
        :param y_seq: (batch_size, output_len, feature_size) (optional)
        :param teacher_forcing_ratio: Probability to use the actual output as the next input
        :return: (batch_size, output_len, output_size)
        '''
        batch_size, trace_len = x_seq.size()
        # x_attr = self.fc_attr_encoder(x_seq_attr).unsqueeze(-1)
        # x_seq_ = self.Conv(x_seq.unsqueeze(-1))
        # x_seq_ = self.positional_encoding(x_seq_)
        x_seq_attr[:, :, -4] = 0
        x_input = torch.cat((x_seq.unsqueeze(-1), x_seq_attr), dim=-1)

        # TimesNet, x_input: [Batchsize, trace_len, n_feature_expended + 1]
        x_input = self.predict_linear(x_input.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(2):
            x_input = self.layer_norm(self.timesnet[i](x_input))
        x_input = self.fc_projection(x_input)

        return x_input[:, -self.output_len:, 0]