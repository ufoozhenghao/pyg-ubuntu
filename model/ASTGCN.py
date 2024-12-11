import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionLayer(nn.Module):
    """
    Compute spatial attention scores
    """
    def __init__(self, num_of_timesteps, num_of_vertices, num_of_features):
        super(SpatialAttentionLayer, self).__init__()
        # self._W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))  #for example (12)
        # self._W2 = nn.Parameter(torch.FloatTensor(num_of_features, num_of_timesteps)) #for example (1, 12)
        # self._W3 = nn.Parameter(torch.FloatTensor(num_of_features)) #for example (1)
        # self._bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices)) #for example (1,307, 307)
        # self._Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices)) #for example (307, 307)
        self._W1 = None
        self._W2 = None
        self._W3 = None
        self._bs = None
        self._Vs = None
        self.initialized = False
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        """
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        if not self.initialized:
            # 根据输入动态初始化参数，但仅在第一次调用时进行
            self._W1 = nn.Parameter(torch.randn(num_of_timesteps, device=x.device))
            self._W2 = nn.Parameter(torch.randn(num_of_features, num_of_timesteps, device=x.device))
            self._W3 = nn.Parameter(torch.randn(num_of_features, device=x.device))
            self._bs = nn.Parameter(torch.randn(1, num_of_vertices, num_of_vertices, device=x.device))
            self._Vs = nn.Parameter(torch.randn(num_of_vertices, num_of_vertices, device=x.device))
            self.initialized = True

        lhs = torch.matmul(torch.matmul(x, self._W1), self._W2)

        rhs = torch.matmul(self._W3, x).transpose(-1, -2)

        product = torch.matmul(lhs, rhs)

        S = torch.matmul(self._Vs, torch.sigmoid(product + self._bs))

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class ChebConvWithSAt(nn.Module):
    """
    K-order Chebyshev graph convolution with Spatial Attention scores
    """
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        """
        super(ChebConvWithSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(K)])
        self.Theta = None

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation

        Chebyshev graph convolution operation
        :param spatial_attention:  shape is (batch_size, N, N) spatial attention scores
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        """
        # in_channels = num_of_features
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        if self.Theta is None:
            self.Theta = nn.Parameter(torch.empty(self.K, num_of_features, self.out_channels, device=x.device))
            nn.init.xavier_uniform_(self.Theta)
        # if self.Theta.nelement() == 0:
        #     self.Theta = nn.Parameter(torch.empty(self.K, num_of_features, self.num_of_filters))
        #     nn.init.xavier_uniform_(self.Theta)

        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step] # (b, N, F_in)
            output = torch.zeros((batch_size, num_of_vertices, self.out_channels), device=x.device)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                T_k_with_at = T_k.mul(spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化
                theta_k = self.Theta[k]   # (in_channel, out_channel)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
                output = output + torch.matmul(rhs, theta_k)
            outputs.append(output.unsqueeze(-1))
        return F.relu(torch.cat(outputs, dim=-1))


class TemporalAttentionLayer(nn.Module):
    """
    Compute temporal attention scores
    len_input=num_of_timesteps
    in_channels=num_of_features
    self.TAt = TemporalAttentionLayer(len_input,num_of_vertices,in_channels)
    """
    def __init__(self,num_of_timesteps, num_of_vertices, num_of_features):
        super(TemporalAttentionLayer, self).__init__()
        self.V_e = None
        self.b_e = None
        self.U_3 = None
        self.U_2 = None
        self.U_1 = None
        self.initialized = False
        # 初始化为空的Parameter
    #     self.U_1 = nn.Parameter(torch.FloatTensor(num_of_vertices))
    #     self.U_2 = nn.Parameter(torch.FloatTensor(num_of_features, num_of_vertices))
    #     self.U_3 = nn.Parameter(torch.FloatTensor(num_of_features))
    #     self.b_e = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps))
    #     self.V_e = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor, x^{(r - 1)}_h
                       shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        E_normalized: torch.Tensor, S', spatial attention scores
                      shape is (batch_size, T_{r-1}, T_{r-1})
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        if not self.initialized:
            # 根据输入动态初始化参数，但仅在第一次调用时进行
            self.U_1 = nn.Parameter(torch.randn(num_of_vertices, device=x.device))
            self.U_2 = nn.Parameter(torch.randn(num_of_features, num_of_vertices, device=x.device))
            self.U_3 = nn.Parameter(torch.randn(num_of_features, device=x.device))
            self.b_e = nn.Parameter(torch.randn(1, num_of_timesteps, num_of_timesteps, device=x.device))
            self.V_e = nn.Parameter(torch.randn(num_of_timesteps, num_of_timesteps, device=x.device))
            self.initialized = True

        print('num_of_vertices:', num_of_vertices) # 38
        print('num_of_features:', num_of_features) # 2   64
        print('num_of_timesteps:', num_of_timesteps) # 5
        print('x.permute(0, 3, 2, 1):', x.permute(0, 3, 2, 1).shape) # torch.Size([32, 5, 2, 38])   torch.Size([32, 5, 64, 38])
        print('self.U_1:', self.U_1.shape) # torch.Size([38])

        _lsh = torch.matmul(x.permute(0, 3, 2, 1), self.U_1)
        print('_lsh:', _lsh.shape) # torch.Size([32, 5, 2])    torch.Size([32, 5, 64])
        print('self.U_2:', self.U_2.shape) # torch.Size([2, 38])
        lhs = torch.matmul(_lsh, self.U_2)  #（32，5，2）x (2,38) = 32,5,38      (32,5,64)x(64,38) = 32,5,38
        print('lhs:', lhs.shape)
        rhs = torch.matmul(self.U_3, x)
        print('rhs:', rhs.shape)
        product = torch.matmul(lhs, rhs)

        E = torch.matmul(self.V_e, torch.sigmoid(product + self.b_e))

        E_normalized = F.softmax(E, dim=1)
        print('========================')
        return E_normalized


class ASTGCNBlock(nn.Module):
    def __init__(self, backbone, len_input=5,num_of_vertices=38,in_channels=2):
        """
        Parameters
        ----------
        backbone: dict, should have 6 keys,
                        "K",
                        "num_of_chev_filters",
                        "num_of_time_filters",
                        "time_conv_kernel_size",
                        "time_conv_strides",
                        "cheb_polynomials"
        """
        super(ASTGCNBlock, self).__init__()

        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
        time_conv_strides = backbone['time_conv_strides']
        cheb_polynomials = backbone["cheb_polynomials"]

        self.SAt = SpatialAttentionLayer(len_input,num_of_vertices,in_channels)
        self.TAt = TemporalAttentionLayer(len_input,num_of_vertices,in_channels)
        # out_channels = num_of_filters = num_of_chev_filters
        self.cheb_conv_SAt = ChebConvWithSAt(
            K=K,
            cheb_polynomials=cheb_polynomials,
            in_channels=in_channels,
            out_channels=num_of_chev_filters
            )
        self.time_conv = nn.Conv2d(
            in_channels=num_of_chev_filters,
            out_channels=num_of_time_filters,
            kernel_size=(1, 3),
            padding=(0, 1),
            stride=(1, time_conv_strides))
        self.residual_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_of_time_filters,
            kernel_size=(1, 1),
            stride=(1, time_conv_strides))
        self.ln = nn.LayerNorm(num_of_time_filters)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor, shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        torch.Tensor, shape is (batch_size, N, num_of_time_filters, T_{r-1})
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # Temporal attention
        temporal_At = self.TAt(x)
        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At)
        x_TAt = x_TAt.reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # Chebyshev GCN with spatial attention
        spatial_At = self.SAt(x_TAt)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)

        # Convolution along time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        print('time_conv_output:', time_conv_output.shape) # torch.Size([32, 64, 38, 5])
        # residual shortcut
        # todo
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)
        print('x_residual:', x_residual.shape)
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual


class ASTGCNSubmodule(nn.Module):
    """
    A submodule in ASTGCN
    """
    def __init__(self, num_for_prediction, backbones, **kwargs):
        """
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        backbones: list(dict), list of backbones
        """
        super(ASTGCNSubmodule, self).__init__(**kwargs)

        self.blocks = nn.Sequential()
        for idx, backbone in enumerate(backbones):
            self.blocks.add_module(f"ASTGCNBlock_{idx}", ASTGCNBlock(backbone))

        # Use convolution to generate the prediction instead of using the fully connected layer
        self.final_conv = nn.Conv2d(
            in_channels=backbones[-1]['num_of_chev_filters'],
            out_channels=num_for_prediction,
            kernel_size=(1, backbones[-1]['num_of_time_filters'])
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor,
           shape is (batch_size, num_of_vertices,
                     num_of_features, num_of_timesteps)

        Returns
        ----------
        torch.Tensor, shape is (batch_size, num_of_vertices, num_for_prediction)
        """
        x = self.blocks(x)
        module_output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return module_output

class ASTGCN(nn.Module):
    """
    ASTGCN, 3 submodules, for hour, day, week respectively
    """
    def __init__(self, num_for_prediction, all_backbones, **kwargs):
        """
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        all_backbones: list[list], 3 backbones for "hour", "day", "week" submodules
        """
        super(ASTGCN, self).__init__(**kwargs)
        if len(all_backbones) <= 0:
            raise ValueError("The length of all_backbones must be greater than 0")

        self.submodules = nn.ModuleList()
        for backbones in all_backbones:
            self.submodules.append(ASTGCNSubmodule(num_for_prediction, backbones))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    def forward(self, x_list):
        """
        Parameters
        ----------
        x_list: list[torch.Tensor], shape is (batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        Returns
        ----------
        Y_hat: torch.Tensor, shape is (batch_size, num_of_vertices, num_for_prediction)
        """
        if len(x_list) != len(self.submodules):
            raise ValueError("Number of submodules does not equal the length of the input list")

        num_of_vertices_set = {x.shape[1] for x in x_list}
        if len(num_of_vertices_set) != 1:
            raise ValueError("Different num_of_vertices detected! Check if your input data have the same size at axis 1.")

        batch_size_set = {x.shape[0] for x in x_list}
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have the same batch size!")

        submodule_outputs = [self.submodules[idx](x_list[idx]) for idx in range(len(x_list))]

        print('submodule_outputs:', submodule_outputs)
        return torch.sum(torch.stack(submodule_outputs), dim=0)


