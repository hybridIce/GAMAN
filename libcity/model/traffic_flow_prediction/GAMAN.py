import math
import numpy as np
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

def calculate_scaled_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    L' = 2L/lambda - I

    Args:
        adj: adj_matrix

    Returns:
        np.ndarray: L'
    """
    n = adj.shape[0]
    d = np.sum(adj, axis=1)  # D
    lap = np.diag(d) - adj     # L=D-A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                lap[i, j] /= np.sqrt(d[i] * d[j])
    lap[np.isinf(lap)] = 0
    lap[np.isnan(lap)] = 0
    lam = np.linalg.eigvals(lap).max().real
    return 2 * lap / lam - np.eye(n)


def calculate_cheb_poly(lap, ks):
    """
    k-order Chebyshev polynomials : T0(L)~Tk(L)
    T0(L)=I/1 T1(L)=L Tk(L)=2LTk-1(L)-Tk-2(L)

    Args:
        lap: scaled laplacian matrix
        ks: k-order

    Returns:
        np.ndarray: T0(L)~Tk(L)
    """
    n = lap.shape[0]
    lap_list = [np.eye(n), lap[:]]
    for i in range(2, ks):
        lap_list.append(np.matmul(2 * lap, lap_list[-1]) - lap_list[-2])
    if ks == 0:
        raise ValueError('Ks must bigger than 0!')
    if ks == 1:
        return np.asarray(lap_list[0:1])  # 1*n*n
    else:
        return np.asarray(lap_list)       # Ks*n*n


def calculate_first_approx(weight):
    '''
    1st-order approximation function.
    :param W: weighted adjacency matrix of G. Not laplacian matrix.
    :return: np.ndarray
    '''
    n = weight.shape[0]
    adj = weight + np.identity(n)
    d = np.sum(adj, axis=1)
    sinvd = np.sqrt(np.linalg.inv(np.diag(d)))
    lap = np.matmul(np.matmul(sinvd, adj), sinvd)  # n*n
    lap = np.expand_dims(lap, axis=0)              # 1*n*n
    return lap
def shortest(adj_mx, type_short_path='hop'):
    """
    Args:
        adj_mx: 邻接矩阵
        type_short_path: 最短路径计算类型（'hop' 或 'dist'）

    Returns:
        np.ndarray: 最短路径矩阵（sh_mx）
    """
    n = adj_mx.shape[0]
    sh_mx = adj_mx.copy()

    if type_short_path == 'hop':
        sh_mx[sh_mx > 0] = 1
        sh_mx[sh_mx == 0] = 511
        for i in range(n):
            sh_mx[i, i] = 0
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    sh_mx[i, j] = min(sh_mx[i, j], sh_mx[i, k] + sh_mx[k, j], 511)
    return sh_mx


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x


class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, drop=0.,
        add_time_in_day=False, add_day_in_week=False, device=torch.device('cpu'),
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)
        if self.add_time_in_day:
            x += self.daytime_embedding((origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long())
        if self.add_day_in_week:
            x += self.weekday_embedding(origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3))
        x = self.dropout(x)
        return x


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1)

    def forward(self, x):  # x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  # return: (batch_size, c_out, input_length-1+1, num_nodes-1+1)


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """

        :param x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        :return: (batch_size, c_out, input_length-kt+1, num_nodes)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]  # (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "GLU":
            # x: (batch_size, c_in, input_length, num_nodes)
            x_conv = self.conv(x)
            # x_conv: (batch_size, c_out * 2, input_length-kt+1, num_nodes)  [P Q]
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            # return P * sigmoid(Q) shape: (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  # residual connection
        return torch.relu(self.conv(x) + x_in)  # residual connection


class TemporalConvLayer_attention(nn.Module):
    def __init__(self, device,  kt, c_in, c_out, num_nodes, dk, changge_dim, heads):
        super(TemporalConvLayer_attention, self).__init__()
        self.kt = kt
        self.c_in = c_in
        self.c_out = c_out
        self.Qdim = changge_dim
        self.dk = dk
        self.head_dim = self.dk//heads
        self.num_heads = heads
        self.num_nodes = num_nodes
        self.align = Align(c_in, c_out)
        self.conv = nn.Conv2d(c_in, 2*c_out, (kt, 1), padding=(1, 0))
        self.change = nn.Conv2d(c_in, self.Qdim, 1)
        self.Wq = nn.Parameter(torch.FloatTensor(self.num_nodes*self.Qdim, self.dk).to(device))
        self.Wqb = nn.Parameter(torch.FloatTensor(self.dk).to(device))
        self.Wk = nn.Parameter(torch.FloatTensor(self.num_nodes*self.Qdim, self.dk).to(device))
        self.Wkb = nn.Parameter(torch.FloatTensor(self.dk).to(device))
        self.b = nn.Parameter(torch.FloatTensor(12).to(device))
        torch.nn.init.kaiming_uniform_(self.Wq)
        torch.nn.init.constant_(self.Wqb, 0)
        torch.nn.init.kaiming_uniform_(self.Wk)
        torch.nn.init.constant_(self.Wkb, 0)
        torch.nn.init.constant_(self.b, 0)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        """
        Args:
            x: (batch_size, F_in, T, N)
        :param x: (batch_size, feature_dim(c_in), T, num_nodes)
        :return: (batch_size, c_out, T, num_nodes)
        """
        x_in = self.align(x)
        B, F, T, N =x.size()
        x_conv = self.conv(x)#(B, 2*F_out, T, N)
        z = self.change(x).permute(0, 2, 1, 3).reshape(B, T, -1)#对输入x进行线性变化，获得(B,T,F*N)
        query = (torch.matmul(z, self.Wq)+self.Wqb).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)#(B, T, qdim)
        key = (torch.matmul(z, self.Wk)+self.Wkb).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)#(B, T, qdim)
        energy = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = torch.relu(energy) ** 2/(self.head_dim*12)
        value = torch.sigmoid(x_conv[:, self.c_out:, :, :]).permute(0, 2, 1, 3).reshape(B, T, self.num_heads, -1).transpose(1, 2)
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).reshape(B, T, F, N).permute(0, 2, 1, 3)
        x = x_conv[:, :self.c_out, :, :] *context
        return self.bn(x)+x_in


class SpatioConvLayer_attention(nn.Module):
    def __init__(self, ks, c_in, c_out, n,  qk_dim, num_heads, geo_mask, device):
        super(SpatioConvLayer_attention, self).__init__()
        # x: (batch_size, F_in, T, N)
        self.ks = ks
        self.num_nodes = n
        self.align = Align(64, 64)
        self.bn = nn.BatchNorm2d(c_out)
        self.geo_mask = geo_mask
        self.c_out = c_out
        self.t_num_heads = num_heads
        self.head_dim =qk_dim//num_heads

        self.geo_q_conv = nn.Conv2d(c_in, qk_dim, kernel_size=1)
        self.geo_k_conv = nn.Conv2d(c_in, qk_dim, kernel_size=1)
        self.geo_v_conv = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.v_head_dim = c_out//num_heads

        self.q_conv = nn.Conv2d(c_in, qk_dim, kernel_size=1)
        self.k_conv = nn.Conv2d(c_in, qk_dim, kernel_size=1)
        self.v_conv = nn.Conv2d(c_in, c_out, kernel_size=1)

        self.scale = self.head_dim**-0.5
        self.proj1 = nn.Conv2d(64, 32, 1)

    def forward(self, x):
        x_in = self.align(x)
        B, D, T, N = x.shape

        x1 = self.proj1(x)#b,f,t,n

        s_geo_q = self.geo_q_conv(x1).permute(0, 2, 3, 1)#(B, T, N, D)
        s_geo_k = self.geo_k_conv(x1).permute(0, 2, 3, 1)
        s_geo_v = self.geo_v_conv(x1).permute(0, 2, 3, 1)
        s_geo_q = s_geo_q.reshape(B, T, N, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)#(B, T, num_heads, N, head_dim)
        s_geo_k = s_geo_k.reshape(B, T, N, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        s_geo_v = s_geo_v.reshape(B, T, N, self.t_num_heads, self.v_head_dim).permute(0, 1, 3, 2, 4)
        s_geo_attn = (s_geo_q @ s_geo_k.transpose(-2, -1))*self.scale#(B, T, num_heads, N, N)

        # 将邻接矩阵为0的位置对应的 s_attn 设为负无穷
        if self.geo_mask is not None:
            s_geo_attn.masked_fill_(self.geo_mask, float('-inf'))
        attention = torch.softmax(s_geo_attn, dim=-1)#(B, T, num_heads, N, N)
        x_geo_v = (attention @ s_geo_v).transpose(1, 2).reshape(B, N, T, -1)
        s_q = self.q_conv(x1).permute(0, 2, 3, 1)#(B, T, N, D)
        s_k = self.k_conv(x1).permute(0, 2, 3, 1)
        s_v = self.v_conv(x1).permute(0, 2, 3, 1)
        s_q = s_q.reshape(B, T, N, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)#(B, T, num_heads, N, head_dim)
        s_k = s_k.reshape(B, T, N, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        s_v = s_v.reshape(B, T, N, self.t_num_heads, self.v_head_dim).permute(0, 1, 3, 2, 4)
        s_attn = (s_q @ s_k.transpose(-2, -1))*self.scale#(B, T, num_heads, N, N)
        attention = torch.softmax(s_attn, dim=-1)#(B, num_heads, N, N)
        x_v = (attention @ s_v).transpose(1, 2).reshape(B, N, T, -1)
        x_v = torch.cat([x_v, x_geo_v], dim =-1).permute(0, 3, 2, 1)
        return torch.relu(x_v+x_in)# [B, 2*dim_out,T, N]


class FullyConvLayer(nn.Module):
    def __init__(self, c, out_dim):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, out_dim, 1)  # c,self.output_dim,1

    def forward(self, x):
        return self.conv(x)


class OutputLayer(nn.Module):
    def __init__(self, c, t, n, out_dim):
        super(OutputLayer, self).__init__()
        self.tconv1 = TemporalConvLayer(t, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = TemporalConvLayer(1, c, c, "sigmoid")  # kernel=1*1
        self.fc = FullyConvLayer(c, out_dim)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)



class STConvBlock_attention(nn.Module):
    def __init__(self, ks, kt, n, c, p,  qk_dim, num_heads, geo_mask, dk, changge_dim, heads,  device):
        super(STConvBlock_attention, self).__init__()

        self.sconv = SpatioConvLayer_attention(ks, 32, 32, n, qk_dim, num_heads, geo_mask, device)
        self.tconv2 = TemporalConvLayer_attention(device, kt, c[1], c[2], n, dk[1], changge_dim[1], heads)
        self.ln2 = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x):  # x: (batch_size, feature_dim/c[0], input_length, num_nodes)
        x = self.sconv(x)
        x = self.tconv2(x)  # (batch_size, c[2], input_length-kt+1-kt+1, num_nodes)
        x_ln = self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


class GAMAN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()

        self.embed_dim = config.get('embed_dim', 64)
        self.ext_dim = self.data_feature.get("ext_dim", 0)

        add_time_in_day = config.get("add_time_in_day", True)
        add_day_in_week = config.get("add_day_in_week", True)

        self.Ks = config.get('Ks', 3)
        self.Kt = config.get('Kt', 3)
        self.blocks = config.get('blocks', [[1, 32, 64], [64, 32, 128]])
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.drop_prob = config.get('dropout', 0)

        self.train_mode = config.get('stgcn_train_mode', 'quick')  # or full
        if self.train_mode.lower() not in ['quick', 'full']:
            raise ValueError('STGCN_train_mode must be `quick` or `full`.')
        self._logger.info('You select {} mode to train STGCN model.'.format(self.train_mode))
        self.blocks[0][0] = self.feature_dim
        if self.input_window - len(self.blocks) * 2 * (self.Kt - 1) <= 0:
            raise ValueError('Input_window must bigger than 4*(Kt-1) for 2 STConvBlock'
                             ' have 4 kt-kernel convolutional layer.')
        self.device = config.get('device', torch.device('cpu'))
        self.graph_conv_type = config.get('graph_conv_type', 'chebconv')
        adj_mx = data_feature['adj_mx']  # ndarray


        self.dk = config.get('dk', [[128, 128], [128, 128]])
        self.changge_dim = config.get('changge_dim', [[2, 2], [2, 2]])
        self.heads = config.get('heads', 1)
        self.adaembed_dim = config.get('adaembed_dim', 128)
        self.qk_dim = config.get('qk_dim', 128)
        self.num_heads = config.get('num_heads', 1)

        self.far_mask_delta = config.get('far_mask_delta', 5)
        sh_mx = shortest(adj_mx)
        sh_mx = sh_mx.T
        self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
        self.geo_mask[sh_mx >= self.far_mask_delta] = 1
        self.geo_mask = self.geo_mask.bool()
        self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
        self.geo_mask[sh_mx >= self.far_mask_delta] = 1
        self.geo_mask = self.geo_mask.bool()

        # 计算GCN邻接矩阵的归一化拉普拉斯矩阵和对应的切比雪夫多项式或一阶近似
        if self.graph_conv_type.lower() == 'chebconv':
            laplacian_mx = calculate_scaled_laplacian(adj_mx)
            self.Lk = calculate_cheb_poly(laplacian_mx, self.Ks)
            self._logger.info('Chebyshev_polynomial_Lk shape: ' + str(self.Lk.shape))
            self.Lk = torch.FloatTensor(self.Lk).to(self.device)
        elif self.graph_conv_type.lower() == 'gcnconv':
            self.Lk = calculate_first_approx(adj_mx)
            self._logger.info('First_approximation_Lk shape: ' + str(self.Lk.shape))
            self.Lk = torch.FloatTensor(self.Lk).to(self.device)
            self.Ks = 1  # 一阶近似保留到K0和K1，但是不是数组形式，只有一个n*n矩阵，所以是1（本质上是2）
        else:
            raise ValueError('Error graph_conv_type, must be chebconv or gcnconv.')

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, self.embed_dim, drop=0,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device,
        )

        # 模型结构
        self.st_conv1 = STConvBlock_attention(self.Ks, self.Kt, self.num_nodes,
                                    self.blocks[0], self.drop_prob, self.qk_dim[0], self.num_heads,  self.geo_mask, self.dk[0],
                                              self.changge_dim[0], self.heads,  self.device)

        self.st_conv2 = STConvBlock_attention(self.Ks, self.Kt, self.num_nodes,
                                    self.blocks[1], self.drop_prob, self.qk_dim[1], self.num_heads,  self.geo_mask, self.dk[1],
                                              self.changge_dim[1], self.heads,  self.device)

        self.output = OutputLayer(self.blocks[1][2], 12, self.num_nodes, self.output_dim)

    def forward(self, batch):
        x = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        x = self.enc_embed_layer(x)
        x = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, input_length, num_nodes)
        x_st1 = self.st_conv1(x)   # (batch_size, c[2](64), input_length-kt+1-kt+1, num_nodes)
        x_st2 = self.st_conv2(x_st1)  # (batch_size, c[2](128), input_length-kt+1-kt+1-kt+1-kt+1, num_nodes)
        outputs = self.output(x_st2)  # (batch_size, output_dim(1), output_length(1), num_nodes)
        outputs = outputs.permute(0, 2, 3, 1)  # (batch_size, output_length(1), num_nodes, output_dim)
        return outputs

    def calculate_loss(self, batch):
        if self.train_mode.lower() == 'quick':
            if self.training:  # 训练使用t+1时间步的loss
                y_true = batch['y'][:, 0:1, :, :]  # (batch_size, 1, num_nodes, feature_dim)
                y_predicted = self.forward(batch)  # (batch_size, 1, num_nodes, output_dim)
            else:  # 其他情况使用全部时间步的loss
                y_true = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
                y_predicted = self.predict(batch)  # (batch_size, output_length, num_nodes, output_dim)
        else:   # 'full'
            y_true = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
            y_predicted = self.predict(batch)  # (batch_size, output_length, num_nodes, output_dim)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)


    def predict(self, batch):
        # 多步预测
        x = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        y = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
        y_preds = []
        x_ = x.clone()
        for i in range(self.output_window):
            batch_tmp = {'X': x_}
            y_ = self.forward(batch_tmp)  # (batch_size, 1, num_nodes, output_dim)
            y_preds.append(y_.clone())
            if y_.shape[-1] < x_.shape[-1]:  # output_dim < feature_dim
                y_ = torch.cat([y_, y[:, i:i+1, :, self.output_dim:]], dim=3)
            x_ = torch.cat([x_[:, 1:, :, :], y_], dim=1)
        y_preds = torch.cat(y_preds, dim=1)  # (batch_size, output_length, num_nodes, output_dim)
        return y_preds
