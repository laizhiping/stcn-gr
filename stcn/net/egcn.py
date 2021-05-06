import torch
import torch.nn as nn


class Gcn(nn.Module):
    def __init__(self, in_channels, out_channels, v):
        super(Gcn, self).__init__()
        self.A = torch.ones(v, v)
        self.I = torch.eye(v, v)
        self.delta = torch.full((v, v), 1e-6)-torch.eye(v,v)*1e-6
        self.delta = nn.Parameter(self.delta)
        self.soft_delta = nn.Softmax(dim=0)
        self.B = nn.Parameter(torch.full((v, v), 1e-6))
        # nn.init.constant_(self.delta, 1e-6)

        self.conv_d = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_e = nn.Conv2d(in_channels, out_channels, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()


    def forward(self, x):
        N, C, T, V = x.size()

        # A = self.I.to(x.device) + self.delta.to(x.device)
        # A = self.relu(A)
        A = self.A.to(x.device)
        # print(A)
        # A = self.I.to(x.device)
        # A = self.soft_delta(A)
        # print(A.requires_grad, self.I.requires_grad, self.delta.requires_grad)

        # y = self.conv_d(torch.matmul(x.view(N, C * T, V), self.B).view(N, C, T, V))

        undigraph = True

        # A.data[A.data>0.8]=1
        if undigraph:
            D = self.get_degree(A.data, -0.5)
            # print(D.requires_grad)
            ADA = torch.mm(D, torch.mm(A, D))
            y = self.conv_d(torch.matmul(x.view(N, C * T, V), ADA+ self.B).view(N, C, T, V))
            # y += self.conv_e(torch.matmul(x.view(N, C * T, V), self.B).view(N, C, T, V))
        else:
            D = self.get_degree(A, -1.0)
            AD = torch.mm(A, D)
            y = self.conv_d(torch.matmul(x.view(N, C * T, V), AD + self.B).view(N, C, T, V))

        # y = self.dropout(y)
        y = self.bn(y)
        y += self.down(x)
        # print(A.shape, y.shape)
        return self.relu(y)

    def get_degree(self, A, r=0.5):
        Dl = torch.sum(A, 0)
        num_node = A.shape[0]
        Dn = torch.zeros((num_node, num_node), device=A.device)
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**r
        return Dn


    def normalize_digraph(self, A):
        Dl = torch.sum(A, 0)
        num_node = A.shape[0]
        Dn = torch.zeros((num_node, num_node), device=A.device)
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        # AD = torch.mm(A, Dn)
        return D

    def normalize_undigraph(self, A):
        Dl = torch.sum(A, 0)
        num_node = A.shape[0]
        Dn = torch.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-0.5)
        # DAD = torch.mm(torch.mm(Dn, A), Dn)
        return DAD

class Tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(x.shape) # [batch*people,channle,frame,point]
        x = self.bn(self.conv(x))
        # print(x.shape)
        # print("#############")
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

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, v, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn = Gcn(in_channels, out_channels, v)
        self.tcn = Tcn(out_channels, out_channels, stride=stride)
        self.se = SE_Block(out_channels, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = Tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        res_x = self.residual(x)
        x = self.gcn(x)
        # print(1, x.shape)
        # print(2, x.shape)
        x = self.tcn(x) + res_x
        # x = self.se(x)
        # x = self.tcn(self.gcn(x)) + self.residual(x)
        # x = self.dropout(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_channels, num_points, num_classes):
        super(Model, self).__init__()

        self.data_bn = nn.BatchNorm1d(num_channels * num_points)
        # self.data_bn = nn.BatchNorm1d(num_channels * 150)

        r = 1
        self.v1 = num_points
        self.v2 = int(self.v1 * r)
        self.v3 = int(self.v2 * r)
        self.v4 = int(self.v3 * r)
# 4,8,8,8
        if self.v1  != num_points:
            # self.avgpool1 = nn.AdaptiveAvgPool1d(self.v1)
            self.avgpool1 = nn.AdaptiveMaxPool1d(self.v1)
        else:
            self.avgpool1 = lambda x: x
        self.l1 = TCN_GCN_unit(num_channels, 4, self.v1, residual=False)

        if self.v2  != self.v1:
            # self.avgpool2 = nn.AdaptiveAvgPool1d(self.v2)
            self.avgpool2 = nn.AdaptiveMaxPool1d(self.v2)
        else:
            self.avgpool2 = lambda x: x
        self.l2 = TCN_GCN_unit(4, 8, self.v2, stride=1)

        if self.v3  != self.v2:
            # self.avgpool3 = nn.AdaptiveAvgPool1d(self.v3)
            self.avgpool3 = nn.AdaptiveMaxPool1d(self.v3)
        else:
            self.avgpool3 = lambda x: x
        self.l3 = TCN_GCN_unit(8, 8, self.v3, stride=1)

        if self.v4  != self.v3:
            # self.avgpool4 = nn.AdaptiveAvgPool1d(self.v4)
            self.avgpool4 = nn.AdaptiveMaxPool1d(self.v4)
        else:
            self.avgpool4 = lambda x: x
        self.l4 = TCN_GCN_unit(8, num_classes, self.v4, stride=1)

        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        N, C, T, V= x.size()  # [batch,channel,frame,point]
        # print(x.shape) # torch.Size([16, 1, 150, 168])
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T) # [batch,point*channel,frame]
        # x = x.view(N, C*T, V)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V) # [batch,channel,frame,point]

        # print(x.shape) # torch.Size([16, 1, 150, 168])

        N, C, T, V = x.size()
        x = x.view(N, C*T, V)
        x = self.avgpool1(x)
        x = x.view(N, C, T, -1)
        x = self.l1(x)
        # print(x.shape) # torch.Size([16, 16, 150, 168])

        N, C, T, V = x.size()
        x = x.view(N, C*T, V)
        x = self.avgpool2(x)
        x = x.view(N, C, T, -1)
        x = self.l2(x)
        # print(x.shape) # torch.Size([16, 64, 75, 134])

        N, C, T, V = x.size()
        x = x.view(N, C*T, V)
        x = self.avgpool3(x)
        x = x.view(N, C, T, -1)
        x = self.l3(x)
        # print(x.shape) # torch.Size([16, 128, 38, 107])

        N, C, T, V = x.size()
        x = x.view(N, C*T, V)
        x = self.avgpool4(x)
        x = x.view(N, C, T, -1)
        x = self.l4(x)
        # print(x.shape) # torch.Size([16, 27, 19, 85])

        # N,C,T,V
        c_new = x.size(1)
        x = x.view(N, c_new, -1)
        # print(x.shape) # torch.Size([16, 27, 1615])
        x = x.mean(2)
        # print(x.shape) # torch.Size([16, 27])

        x = self.dropout(x)
        return x

if __name__ == "__main__":
    net = Model(num_channels=1, num_points=168, num_classes=27)
    net.cuda()
    x = torch.randn(16,1,150,168).cuda()
    from thop import profile
    # flops, params = profile(net, inputs=(x, ))
    # print(flops, params)
    # # print(x.shape)
    y = net(x)
    print(y.shape)