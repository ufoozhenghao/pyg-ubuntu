import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionLayer(nn.Module):
    """
    Compute spatial attention scores
    """
    def __init__(self):
        super(SpatialAttentionLayer, self).__init__()
        self.W_1 = nn.Parameter(torch.Tensor())
        self.W_2 = nn.Parameter(torch.Tensor())
        self.W_3 = nn.Parameter(torch.Tensor())
        self.b_s = nn.Parameter(torch.Tensor())
        self.V_s = nn.Parameter(torch.Tensor())

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor, x^{(r - 1)}_h,
           shape is (batch_size, N, C_{r-1}, T_{r-1})

        Returns
        ----------
        S_normalized: torch.Tensor, S', spatial attention scores
                      shape is (batch_size, N, N)
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # Initialize parameters
        if self.W_1.nelement() == 0:
            self.W_1 = nn.Parameter(torch.empty(num_of_timesteps))
            self.W_2 = nn.Parameter(torch.empty(num_of_features, num_of_timesteps))
            self.W_3 = nn.Parameter(torch.empty(num_of_features))
            self.b_s = nn.Parameter(torch.empty(1, num_of_vertices, num_of_vertices))
            self.V_s = nn.Parameter(torch.empty(num_of_vertices, num_of_vertices))
            nn.init.xavier_uniform_(self.W_1)
            nn.init.xavier_uniform_(self.W_2)
            nn.init.xavier_uniform_(self.W_3)
            nn.init.zeros_(self.b_s)
            nn.init.xavier_uniform_(self.V_s)

        lhs = torch.matmul(torch.matmul(x, self.W_1), self.W_2)

        rhs = torch.matmul(self.W_3, x.permute(2, 0, 3, 1))

        product = torch.matmul(lhs, rhs)

        S = torch.matmul(self.V_s, torch.sigmoid(product + self.b_s).permute(1, 2, 0)).permute(2, 0, 1)

        S = S - torch.max(S, dim=1, keepdim=True)[0]
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, dim=1, keepdim=True)
        return S_normalized


class ChebConvWithSAt(nn.Module):
    """
    K-order Chebyshev graph convolution with Spatial Attention scores
    """
    def __init__(self, num_of_filters, K, cheb_polynomials):
        super(ChebConvWithSAt, self).__init__()
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        self.Theta = nn.Parameter(torch.Tensor())

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: torch.Tensor, graph signal matrix
           shape is (batch_size, N, F, T_{r-1}), F is the num of features

        spatial_attention: torch.Tensor, shape is (batch_size, N, N)
                           spatial attention scores

        Returns
        ----------
        torch.Tensor, shape is (batch_size, N, self.num_of_filters, T_{r-1})
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        if self.Theta.nelement() == 0:
            self.Theta = nn.Parameter(torch.empty(self.K, num_of_features, self.num_of_filters))
            nn.init.xavier_uniform_(self.Theta)

        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros((batch_size, num_of_vertices, self.num_of_filters), device=x.device)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k]
                T_k_with_at = T_k * spatial_attention
                theta_k = self.Theta[k]
                rhs = torch.matmul(T_k_with_at.permute(0, 2, 1), graph_signal)
                output = output + torch.matmul(rhs, theta_k)
            outputs.append(output.unsqueeze(-1))
        return F.relu(torch.cat(outputs, dim=-1))


class TemporalAttentionLayer(nn.Module):
    """
    Compute temporal attention scores
    """
    def __init__(self):
        super(TemporalAttentionLayer, self).__init__()
        self.U_1 = nn.Parameter(torch.Tensor())
        self.U_2 = nn.Parameter(torch.Tensor())
        self.U_3 = nn.Parameter(torch.Tensor())
        self.b_e = nn.Parameter(torch.Tensor())
        self.V_e = nn.Parameter(torch.Tensor())

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

        if self.U_1.nelement() == 0:
            self.U_1 = nn.Parameter(torch.empty(num_of_vertices))
            self.U_2 = nn.Parameter(torch.empty(num_of_features, num_of_vertices))
            self.U_3 = nn.Parameter(torch.empty(num_of_features))
            self.b_e = nn.Parameter(torch.empty(1, num_of_timesteps, num_of_timesteps))
            self.V_e = nn.Parameter(torch.empty(num_of_timesteps, num_of_timesteps))
            nn.init.xavier_uniform_(self.U_1)
            nn.init.xavier_uniform_(self.U_2)
            nn.init.xavier_uniform_(self.U_3)
            nn.init.zeros_(self.b_e)
            nn.init.xavier_uniform_(self.V_e)

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U_1), self.U_2)

        rhs = torch.matmul(self.U_3, x.permute(2, 0, 1, 3))

        product = torch.matmul(lhs, rhs)

        E = torch.matmul(self.V_e, torch.sigmoid(product + self.b_e).permute(1, 2, 0)).permute(2, 0, 1)

        E = E - torch.max(E, dim=1, keepdim=True)[0]
        exp = torch.exp(E)
        E_normalized = exp / torch.sum(exp, dim=1, keepdim=True)
        return E_normalized


class ASTGCNBlock(nn.Module):
    def __init__(self, backbone):
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

        self.SAt = SpatialAttentionLayer()
        self.cheb_conv_SAt = ChebConvWithSAt(
            num_of_filters=num_of_chev_filters,
            K=K,
            cheb_polynomials=cheb_polynomials)
        self.TAt = TemporalAttentionLayer()
        self.time_conv = nn.Conv2d(
            in_channels=num_of_chev_filters,
            out_channels=num_of_time_filters,
            kernel_size=(1, 3),
            padding=(0, 1),
            stride=(1, time_conv_strides))
        self.residual_conv = nn.Conv2d(
            in_channels=num_of_chev_filters,
            out_channels=num_of_time_filters,
            kernel_size=(1, 1),
            stride=(1, time_conv_strides))
        self.ln = nn.LayerNorm(num_of_time_filters)

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
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        # Residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

        return self.ln(F.relu(x_residual + time_conv_output))


class ASTGCNSubmodule(nn.Module):
    """
    A submodule in ASTGCN
    """
    def __init__(self, num_for_prediction, backbones):
        """
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        backbones: list(dict), list of backbones
        """
        super(ASTGCNSubmodule, self).__init__()

        self.blocks = nn.Sequential()
        for backbone in backbones:
            self.blocks.add_module("ASTGCNBlock", ASTGCNBlock(backbone))

        # Use convolution to generate the prediction instead of using the fully connected layer
        self.final_conv = nn.Conv2d(
            in_channels=backbones[-1]['num_of_time_filters'],
            out_channels=num_for_prediction,
            kernel_size=(1, 1)
        )
        self.W = nn.Parameter(torch.Tensor(backbones[-1]['num_of_vertices'], num_for_prediction))
        self.reset_parameters()

    # 用于初始化或重置模型的参数
    def reset_parameters(self):
        """
        用于对参数进行 Xavier 均匀初始化。Xavier 初始化（也称为 Glorot 初始化）是一种常用的权重初始化方法，
    旨在保持输入和输出的方差相同，从而避免梯度消失或爆炸的问题
        self.W 是一个 nn.Parameter 对象，表示模型中的一个可训练参数。
    在这个具体的例子中，它是一个形状为 (num_of_vertices, num_for_prediction) 的权重矩阵，用于对模型的最后输出进行线性变换
        """
        nn.init.xavier_uniform_(self.W)

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
        return module_output * self.W

class ASTGCN(nn.Module):
    """
    ASTGCN, 3 submodules, for hour, day, week respectively
    """
    def __init__(self, num_for_prediction, all_backbones):
        """
        Parameters
        ----------
        num_for_prediction: int, how many time steps will be forecasting

        all_backbones: list[list], 3 backbones for "hour", "day", "week" submodules
        """
        super(ASTGCN, self).__init__()
        if len(all_backbones) <= 0:
            raise ValueError("The length of all_backbones must be greater than 0")

        self.submodules = nn.ModuleList()
        for backbones in all_backbones:
            self.submodules.append(ASTGCNSubmodule(num_for_prediction, backbones))

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

        return torch.sum(torch.stack(submodule_outputs), dim=0)


