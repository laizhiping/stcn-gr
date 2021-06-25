import torch
import torch.nn as nn
import math

def conv_branch_init(conv):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, v):
        super(GCN, self).__init__()
        self.A = torch.ones(v, v)
        self.B = nn.Parameter(torch.full((v, v), 1e-6))

        self.w = nn.Conv2d(in_channels, out_channels, 1)

        if in_channels != out_channels:
            self.rb = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.rb = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        conv_branch_init(self.w)

    def forward(self, x):
        N, C, T, V = x.size()

        A = self.A.to(x.device)

        undigraph = True
        if undigraph:
            D = self.get_degree(A.data, -0.5)
            ADA = torch.mm(D, torch.mm(A, D))
            y = self.w(torch.matmul(x.view(N, C * T, V), ADA+ self.B).view(N, C, T, V))
        else:
            D = self.get_degree(A, -1.0)
            AD = torch.mm(A, D)
            y = self.w(torch.matmul(x.view(N, C * T, V), AD + self.B).view(N, C, T, V))

        y = self.bn(y)
        y += self.rb(x)

        return self.relu(y)

    def get_degree(self, A, r=0.5):
        D = torch.sum(A, 0)
        num_node = A.shape[0]
        Dn = torch.zeros((num_node, num_node), device=A.device)
        for i in range(num_node):
            if D[i] > 0:
                Dn[i, i] = D[i]**r
        return Dn


class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TCN, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        conv_init(self.conv)
        bn_init(self.bn, 1)

    # batch, channel, frame, point
    def forward(self, x):
        # B, C2, T, V
        x = self.conv(x)
        x = self.bn(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=2):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上

class STCB(nn.Module):
    def __init__(self, in_channels, out_channels, v, stride=1, residual=True):
        super(STCB, self).__init__()
        self.gcn = GCN(in_channels, out_channels, v)
        self.tcn = TCN(out_channels, out_channels, stride=stride)
        # self.se = SE_Block(out_channels, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        if not residual:
            self.rb = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.rb = lambda x: x
        else:
            self.rb = TCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        res_x = self.rb(x)

        x = self.gcn(x)
        x = self.tcn(x) + res_x
        # x = self.se(x)
        x = self.relu(x)
        
        return x


class STCN(nn.Module):
    def __init__(self, num_channels, num_points, num_classes):
        super(STCN, self).__init__()
        self.data_bn = nn.BatchNorm1d(num_channels * num_points)
        self.stcbs = nn.Sequential(
            STCB(num_channels, 4, num_points, residual=False),
            STCB(4, 8, num_points, stride=1),
            STCB(8, 8, num_points, stride=1),
            STCB(8, num_classes, num_points, stride=1)
        )
        self.dropout = nn.Dropout(0.5)

        bn_init(self.data_bn, 1)

    # batch, channel, frame, point
    def forward(self, x):
        N, C, T, V = x.size()
        # N, V*C, T
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        # N, C, T, V
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)

        # N, C2, T, V
        x = self.stcbs(x)

        C2 = x.size(1)
        # N, C2, T*V
        x = x.view(N, C2, -1)
        # N, C2
        x = x.mean(2)

        # x = self.dropout(x)
        return x
