from __future__ import division
import argparse
import math
import time
import random
import os
import copy
import numpy as np
from torch import optim
import torch.nn.functional as F
from utils import load_metr_la_rdata, calculate_random_walk_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
#from thop import profile

def parse_arg():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='metr',help='name of the datasets,select from metr,nrel,ushcn,sedata of pemsbay')
    parser.add_argument('--n_s',type=int,default=104,help='sampled space dimension')  # n_s=n_o+n_m
    parser.add_argument('--h',type=int,default=25,help='sampled time dimension')    # 窗口大小，这里和dualstn一致为25
    parser.add_argument('--z',type=int,default=100,help='hidden dimension for graph convolution')   #没有用了
    parser.add_argument('--K',type=int,default=1,help='if use diffusion convolution,the actual diffusion conv step is K+1') #无用
    parser.add_argument('--n_m',type=int,default=52,help='number of mask node during training') #掩码点数量
    parser.add_argument('--n_u',type=int,default=103,help='target locations,n_u locations will be deleted from training data')  #未知点数量
    parser.add_argument('--epochs',type=int,default=200,help='max training episode')
    parser.add_argument('--learning_rate',type=float,default=0.0005,help='the learning rate')
    parser.add_argument('--E_maxvalue',type=int,default=80,help='the max value from experience')    #无用
    parser.add_argument('--batch_size',type=int,default=4,help='batch_size')
    parser.add_argument('--to_plot',type=bool,default=True,help='Whether to plot the RMSE training result')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--id', type=int, default=1, help='id')     #id和种子保持一致，1-10最好，方便记录
    parser.add_argument('--gpu_id',type=int,default=0,help='which gpu 0 or 1')
    parser.add_argument('--blocks',type=int,default=3)      #可以改
    parser.add_argument('--layers',type=int,default=2)      #可以改
    parser.add_argument('--dropout',type=int,default=0.1)   #可以改
    parser.add_argument('--residual',type=int,default=32,help='the channels of residual')   #可以
    parser.add_argument('--dilation', type=int, default=32, help='the channels of dilation')    #可以
    parser.add_argument('--skip',type=int,default=256)  #可以
    parser.add_argument('--end',type=int,default=256)   #可以
    parser.add_argument('--patience',type=int,default=30)   #earlystop
    parser.add_argument('--aux_h',type=int,default=100) #无用
    parser.add_argument('--aux_layer',type=int,default=2)   #无用
    parser.add_argument('--gid', type=int, default=808)   #组id，会根据gid生成对应的log文件，如1.log，其中通过id来区分是哪个种子

    args=parser.parse_args()
    return args

class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """
    def __init__(self, in_channels, out_channels, orders=1, activation = 'relu'):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1         #这是什么东西？
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                             out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)          #用U（-stdv,stdv)的均匀分布填充theta1,作为重置
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)          #重置偏置bias

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0] # batch_size
        num_node = X.shape[1]
        input_size = X.size(2)  # time_length       #和in_channels一个东西
        supports = []
        supports.append(A_q)
        supports.append(A_h)

        x0 = X.permute(1, 2, 0) #(num_nodes, num_times, batch_size)   交换维度
        x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])   #input_size也即num_timesteps
        x = torch.unsqueeze(x0, 0)          # x.shape=[1，num_node,input_size*batch_size]
        for support in supports:            # support.shape=[num_nodes,num_nodes]
            x1 = torch.mm(support, x0)      #x1.shape=[num_nodes,input_size*batch_size]
            x = self._concat(x, x1)         #在第0维拼接，每循环一次x.shape[0]++ x.shape=[2,num_nodes,input_size*batch_size],注意_concat会给第二个补维度
            for k in range(2, self.orders + 1):         #x.shape=[2*orders+1,num_nodes,input_size*batch_size],注意_concat会给第二个补维度
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)    Theta1.shape=[input_size*matrics,output]
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)

        return x

class nconv1(nn.Module):
    def __init__(self):
        super(nconv1,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,cw->nwvl',(x,A)) # b,2,207,12 * 207,207---->b,2,207,12
        return x.contiguous()
class gcn1(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super(gcn1, self).__init__()
        self.nconv = nconv1()
        c_in = (order * support_len + 1) * c_in  # 一共有这么多个x
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)  # 64,160,207,12
        h = self.mlp(h)  # 64,32,207,12
        h = F.dropout(h, self.dropout, training=self.training)
        return h
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A)) # b,2,207,12 * 207,207---->b,2,207,12
        return x.contiguous()

class linear(nn.Module):    #使用1×1卷积来作为mlp
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)
class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in   # 一共有这么多个x
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)    # 64,160,207,12
        h = self.mlp(h)             # 64,32,207,12
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DGL2(nn.Module):   #期望输入bfnt，输出nn
    def __init__(self,time_dim,feature,batchsize):
        super(DGL2,self).__init__()
        self.d_k=32                         # 可以改
        self.dim=time_dim*feature
        self.W_Q=nn.Linear(self.dim,self.d_k,bias=False)
        self.W_K=nn.Linear(self.dim,self.d_k,bias=False)
        self.a = torch.nn.Parameter(torch.randn(batchsize,1,time_dim*feature))
        self.mlp = torch.nn.Conv1d(time_dim, time_dim, kernel_size=1, bias=True)
        self.importance = nn.Linear(time_dim * feature,time_dim * feature)
        self.conv = torch.nn.Conv2d(feature,feature,kernel_size=(1,5))


    def forward(self,x):    # bfnt
        residual, batch_size, feature, num_nodes = x, x.size(0), x.size(1), x.size(2)
        x=x.permute(0,2,3,1)    #bntf
        # x2 = (self.conv(x[:,:,:5,:].permute(0,3,1,2))).permute(0,2,3,1)
        x=torch.reshape(x,shape=(batch_size,num_nodes,-1))
        #x2 = torch.reshape(x2, shape=(batch_size, num_nodes, -1))
        importance = self.importance(self.a)
        importance[importance<0.2]=0
        x1 = x*importance
        # for i in range(batch_size):
        #     x1 = x1.clone()  # 创建x1的副本，避免原地操作
        #     x1[i,:,:] = x1[i,:,:] * importance[i,:]
        #x1 = x1*importance
        Q=self.W_Q(x1)
        K=self.W_K(x)
        q_mean=torch.mean(Q,dim=-1,keepdim=True)
        q_std=torch.std(Q,dim=-1,keepdim=True)
        k_mean=torch.mean(K,dim=-1,keepdim=True)
        k_std=torch.std(K,dim=-1,keepdim=True)
        Q=(Q-q_mean)/(q_std+1e-6)
        K=(K-k_mean)/(k_std+1e-6)
        scores=torch.matmul(Q.transpose(-1,-2),K)/np.sqrt(self.d_k)
        attn=nn.Softmax(dim=-1)(torch.sum(scores,dim=0))
        return attn
class DGL(nn.Module):   #期望输入bfnt，输出nn
    def __init__(self,time_dim,feature):
        super(DGL,self).__init__()
        self.d_k=32                         # 可以改
        self.dim=time_dim*feature
        self.W_Q=nn.Linear(self.dim,self.d_k,bias=False)
        self.W_K=nn.Linear(self.dim,self.d_k,bias=False)

    def forward(self,x):    # bfnt
        residual, batch_size, feature, num_nodes = x, x.size(0), x.size(1), x.size(2)
        x=x.permute(0,2,3,1)    #bntf
        x=torch.reshape(x,shape=(batch_size,num_nodes,-1))
        Q=self.W_Q(x)
        K=self.W_K(x)
        q_mean=torch.mean(Q,dim=-1,keepdim=True)
        q_std=torch.std(Q,dim=-1,keepdim=True)
        k_mean=torch.mean(K,dim=-1,keepdim=True)
        k_std=torch.std(K,dim=-1,keepdim=True)
        Q=(Q-q_mean)/(q_std+1e-6)
        K=(K-k_mean)/(k_std+1e-6)
        scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(self.d_k)
        attn=nn.Softmax(dim=-1)(torch.sum(scores,dim=0))
        return attn

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qfc=nn.Linear(2,64,bias=True)
        self.kfc=nn.Linear(2,64,bias=True)
        self.vfc=nn.Linear(2,64,bias=True)
        self.d=64
        self.li=nn.Linear(64,1)
    def forward(self,x,):   #btn2
        q=self.qfc(x)
        k=self.kfc(x)
        v=self.vfc(x)
        att=torch.matmul(q,k.transpose(-1,-2))/np.sqrt(self.d)
        att=torch.softmax(att,dim=-1)
        res=torch.matmul(att,v)     #btn64
        res=self.li(res).squeeze(-1)
        return res

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, num_of_d):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.num_of_d =num_of_d

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, F,n_heads, len_q, d_k]
        K: [batch_size, F,n_heads, len_k, d_k]
        V: [batch_size, F,n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        attn = F.softmax(scores, dim=3)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, scores
class MultiHeadAttention(nn.Module):
    def __init__(self,in_dim,channel, d_k=32 ,d_v=32, n_heads=3):  #in_dim是时间维度
        super(MultiHeadAttention, self).__init__()
        self.in_dim = in_dim
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.channel = channel
        self.W_Q = nn.Linear(in_dim, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(in_dim, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(in_dim, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, in_dim, bias=False)

    def forward(self, input_Q, input_K, input_V):
        '''
        输入=[B,F,N,T]
        '''
        residual, batch_size = input_Q, input_Q.size(0) #输入[32,1,n,t]
        Q = self.W_Q(input_Q).view(batch_size, self.channel, -1, self.n_heads, self.d_k).transpose(2, 3)  # ,这里计算空间注意力，把时间当维度，由t映射到3*32，f不动
        K = self.W_K(input_K).view(batch_size, self.channel, -1, self.n_heads, self.d_k).transpose(2, 3)  # K: [batch_size, F,n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, self.channel, -1, self.n_heads, self.d_v).transpose(2, 3)  # V: [batch_size, F,n_heads, len_v(=len_k), d_v]
        context, res_attn = ScaledDotProductAttention(self.d_k, self.channel)(Q, K, V)
        context = context.transpose(2, 3).reshape(batch_size, self.channel, -1, self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # 还原为原来时间维度
        return nn.LayerNorm(self.in_dim).to(device)(output + residual) #有残差说明是时间维度的注意力

class EstimationGate(nn.Module):
    """The estimation gate module."""

    def __init__(self, time_emb_dim, hidden_dim): # 10,10,64
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(time_emb_dim * 2, hidden_dim)   #40,64
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(hidden_dim, 1) #64,1

    def forward(self, time_in_day_feat, day_in_week_feat, history_data):#输入btnf，，输出btnf，这里x是bfnt
        """Generate gate value in (0, 1) based on current node and time step embeddings to roughly estimating the proportion of the two hidden time series."""

        history_data=history_data.permute(0,3,2,1)  #转为btnf
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        # 将这四个张量在f维拼接，时间嵌入【b,12,207,10]不用改变了，结点嵌入[207,10]需要广播一下。
        estimation_gate_feat = torch.cat([time_in_day_feat, day_in_week_feat], dim=-1)
        hidden = self.fully_connected_layer_1(estimation_gate_feat) #b,12,207,40-->b,12,207,64,在f维度上拼接
        hidden = self.activation(hidden)    #激活一下
        # activation
        estimation_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))[:, -history_data.shape[1]:, :, :] #2,12,207,1,先变换再sigmoid再统一时间维度，
        history_data = history_data * estimation_gate # 2,12,207,32 * 2,12,207,1，相乘，得到过滤出的数据
        history_data=history_data.permute(0,3,2,1)  #转为bfnt
        return history_data

class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class gwnet(nn.Module): #输入bnt,->b1nt->b32nt->b32nt'->b256nt''-->b512nt'''->b24n1-->b1n24->bnt,输出bnt
    def __init__(self, device,dropout=0.1,  in_dim=1,out_dim=25,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=3,layers=2,aux_h=100,aux_layer=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.pointwise1=nn.ModuleList()
        self.pointwise2 = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dgl=nn.ModuleList()
        self.dgl2 = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.chomp=nn.ModuleList()


        self.T_i_D_emb  = nn.Parameter(torch.empty(288, 10)).to(device)    #288,10的空张量
        self.D_i_W_emb  = nn.Parameter(torch.empty(7, 10)) .to(device) #7,10的空张量
        # node embeddings
        self.node_emb_u = nn.Parameter(torch.empty(207, 10))    #207,10
        self.node_emb_d = nn.Parameter(torch.empty(207, 10))    #207，10


        self.start_gcn=nn.ModuleList()
        self.aux_layer=aux_layer
        self.aux_h=aux_h
        self.start_gcn.append(D_GCN(in_channels=out_dim,out_channels=aux_h))
        for i in range(1,self.aux_layer-1):
            self.start_gcn.append(D_GCN(in_channels=aux_h,out_channels=aux_h))
        self.start_gcn.append(D_GCN(in_channels=aux_h,out_channels=out_dim))

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels,  kernel_size=(1,1))
        input_len=out_dim
        receptive_field = 1
        self.supports_len = 2
        self.total_time_len=0
        self.se=nn.ModuleList()
        self.estimation_gate=nn.ModuleList()
        self.attention=nn.ModuleList()
        self.fc=nn.ModuleList()
        self.bias_conv=nn.ModuleList()
        self.fcp=nn.ModuleList()
        self.fcq=nn.ModuleList()
        self.fc1 = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.conv2d_trans = nn.ModuleList()
        self.linearx = nn.ModuleList()

        for b in range(blocks): #开始构建模型，一个block接一个block
            additional_scope = kernel_size - 1  #每个block包含空洞卷积，需要考虑padding操作
            new_dilation = 1    # 每个block开始的空洞是1
            for i in range(layers): # 每个block里卷积不只一次，即层数不只1
                # dilated convolutions
                padding = (kernel_size - 1) * new_dilation
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=residual_channels,
                                                   kernel_size=(1,kernel_size),dilation=(1,new_dilation),padding=(0,padding),groups=residual_channels))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, kernel_size), dilation=(1,new_dilation),padding=(0,padding),groups=residual_channels))

                self.pointwise1.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1,1), padding=0,stride=1))
                self.pointwise2.append(
                    nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, 1), padding=0, stride=1))

                self.bias_conv.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=(1,new_dilation),padding=(0,padding)))

                self.chomp.append(Chomp1d(padding))

                self.conv.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=residual_channels,
                                                   kernel_size=(1,kernel_size),dilation=(1,2*new_dilation),padding=(0,padding),groups=residual_channels))

                self.conv1.append(nn.Conv2d(in_channels=residual_channels,
                                           out_channels=residual_channels,
                                           kernel_size=(1, kernel_size), dilation=(1, 2 * new_dilation),
                                           padding=(0, padding), groups=residual_channels))
                self.linearx.append(nn.Linear(h*2,h))

                self.conv2d_trans.append(nn.ConvTranspose2d(in_channels=residual_channels,
                                                   out_channels=residual_channels,
                                                   kernel_size=(1,kernel_size),dilation=(1,2*new_dilation),padding=(0,padding),groups=residual_channels))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                self.se.append(SEnet(dilation_channels))

                self.estimation_gate.append(EstimationGate(time_emb_dim=10, hidden_dim=64))

                self.gconv.append(gcn(dilation_channels,residual_channels,dropout))
                self.gconv2.append(gcn1(dilation_channels, residual_channels, dropout))
                self.fc.append(nn.Conv2d(in_channels=2 * dilation_channels, out_channels=dilation_channels,kernel_size=(1,1)))  #在f维拼接
                self.fcp.append(nn.Linear(out_dim,out_dim))  # 在f维拼接
                self.fcq.append(nn.Linear(out_dim,out_dim))  # 在f维拼接
                self.fc1.append(nn.Conv2d(in_channels=2 * dilation_channels, out_channels=dilation_channels,
                                          kernel_size=(1, 1)))  # 在f维拼接

        for b in range(blocks):
            additional_scope=kernel_size-1
            for i in range(layers):
                # input_len-=additional_scope
                additional_scope*=2
                self.dgl.append(DGL(time_dim=input_len,feature=dilation_channels))
                self.dgl2.append(DGL2(time_dim=input_len, feature=dilation_channels, batchsize=4))
                self.attention.append(MultiHeadAttention(in_dim=input_len,channel=dilation_channels))
                # self.fc.append(nn.Linear(in_features=2*input_len,out_features=input_len,bias=False))      #在t维拼接

                self.total_time_len=self.total_time_len+input_len
        self.end_conv_1 = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=skip_channels,out_channels=out_dim,kernel_size=(1,1), bias=True)
        self.end_conv_t=nn.Linear(self.total_time_len,1)
        self.receptive_field = receptive_field


    def onehot(self, x):    #btnf=3
        tod = F.one_hot(x[:, :, :, 1].long(), num_classes=288)  # b,12,307,288
        dow = F.one_hot(x[:, :, :, 2].long(), num_classes=7)    #b,12,307,7]
        x = torch.cat((tod, dow), dim=-1) * x[:, :, :, :1]  #[b,12,307,295]
        return x

    def _prepare_inputs(self, history_data): # btnf
        num_feat    = 1
        # node embeddings
        node_emb_u  = self.node_emb_u  # [N, d] 把申请好的207,10赋值给Node_emb_u
        node_emb_d  = self.node_emb_d  # [N, d] 一样，207,10
        # time slot embedding
        time_in_day_feat = self.T_i_D_emb[(history_data[:, :, :, num_feat] * 288).type(torch.LongTensor)]    # 输入中的维度1是tid信息，拿出来进行嵌入，[B, L, N, d] b 12 207 10
        day_in_week_feat = self.D_i_W_emb[(history_data[:, :, :, num_feat+1]).type(torch.LongTensor)]          # [B, L, N, d] b 12 207 10
        # traffic signals
        history_data = history_data[:, :, :, :num_feat] #只保留流量数据，数据和提示信息分离，

        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def forward(self, input,A_q,A_h):   # btn3
        input, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat = self._prepare_inputs(input)
        # input=self.onehot(input)    #btn295
        input=input.permute(0,3,2,1)    #b3nt
        in_len = input.size(3)  # 取出时间维度，这里在第四维
        if in_len<self.receptive_field: #当时间长度小于感受野，就要先pad
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)  #充当开始的FC,b1nt-->b32nt
        skip = []
        Adlist=[]
        # WaveNet layers
        for i in range(self.blocks * self.layers):


            scale = []
            residual1 = x  # b32nt
            filter = self.conv[i](residual1)
            filter = torch.tanh(filter)
            gate = self.conv1[i](residual1)
            gate = torch.sigmoid(gate)
            x1 = filter * gate
            x1 = self.se[i](x1)
            scale.append(x1)


            # G-TCN
            residual = x
            filter = self.filter_convs[i](residual)
            filter = self.chomp[i](filter)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = self.chomp[i](gate)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = self.se[i](x)
            scale.append(x)
            x = torch.cat(scale,dim=-1)
            x = self.linearx[i](x)


            # Skip-connection
            s = x
            skip.append(s)

            supports = []
            supports.append(A_q)
            supports.append(A_h)
            A_d=self.dgl[i](x)
            Adlist.append(A_d)
            supports.append(A_d)
            #分解成两部分
            # x_dif=self.estimation_gate[i](time_in_day_feat,day_in_week_feat,x)
            # x_inh=x-x_dif
            x_dif=x
            # x_inh=x
            #图卷积
            x_dif1 = self.gconv[i](x_dif, supports)
            x_dif1 = torch.tanh(x_dif1)
            x_dif2 = self.gconv[i](x_dif, supports)
            x_dif2 = torch.sigmoid(x_dif2)
            x_dif = x_dif1 * x_dif2
            A_f = self.dgl2[i](x_dif)
            x_f = self.gconv2[i](x_dif, [A_f, A_f.T])
            x_dif = self.fc1[i]((torch.concat((x_dif, x_f), dim=1)))
            x_inh = x - x_dif
            #注意力
            x2=self.attention[i](x_inh,x_inh,x_inh)     #输入多少，输出维度就是多少，要求输入bfnt
            x=torch.cat((x_dif,x2),dim=1)
            x=self.fc[i](x)
            # x=x1+x2
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        skip=torch.cat(skip,dim=-1)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  #b 24 n 12+11+...
        x=self.end_conv_t(x)    #b t n 1
        x=torch.squeeze(x,-1)
        Adlist=torch.stack(Adlist,dim=0)
        return x,Adlist,A_f

class SEnet(nn.Module):
    def __init__(self,channels,ratio=4):
        super(SEnet, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        # 经过两次全连接层，一次较小，一次还原
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio,False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_=x.size() #取出batch size和通道数
        # b,c,w,h->b,c,1,1->b,c 以便进行全连接
        avg=self.avgpool(x).view(b,c)
        #b,c->b,c->b,c,1,1 以便进行线性加权
        fc=self.fc(avg).view(b,c,1,1)
        return fc*x
#***************************************************************************************************************************************
def load_data(dataset):
    '''Load dataset
    Input: dataset name
    Returns
    -------
    A: adjacency matrix
    X: processed data
    capacity: only works for NREL, each station's capacity
    '''
    capacity = []
    tod=True
    dow=True
    if dataset == 'metr':
        A, X = load_metr_la_rdata()
        X = X[:, 0:1, :]    #n 1 t
    # x=ntf
    X=X.transpose((2,0,1))    #tnf
    if tod:
        MAX_TOD = 288   #288
        tod = [(i % MAX_TOD) for i in range(X.shape[0])] #list:16992,把数据片的序号记录

        tod = [t / MAX_TOD for t in tod]
        tod = np.tile(tod, [1, X.shape[1], 1]).transpose((2, 1, 0))  # [16992,307,1],对应原始数据
        data = np.concatenate((X, tod), axis=-1)
    if dow:
        MAX_TOD = 288   #288
        MAX_DOW = 7
        dow = [((i // MAX_TOD) % MAX_DOW) for i in range(data.shape[0])] #list:16992,对应每个时间片在哪周
        dow = np.tile(dow, [1, data.shape[1], 1]).transpose((2, 1, 0))  #16992,307,1
        data = np.concatenate((data, dow), axis=-1) #【16992,307,3】
    X=data.transpose(1,0,2) #ntf
    split_line1 = int(X.shape[1] * 0.7)
    # num_val=int(X.shape[1]*0.2)
    num_test=int(X.shape[1]-split_line1)

    training_set = X[:, :split_line1,:].transpose(1,0,2)
    mean=training_set[:,:,0].mean()
    std=training_set[:,:,0].std()
    print('training_set', training_set.shape)
    # val_set=X[:,split_line1:split_line1+num_val,:].transpose(1,0,2)
    test_set = X[:, -num_test:,:].transpose(1,0,2)  # split the training and test period

    test_node=np.load(f'data/{dataset}/testnode.npz')
    unknow_set=test_node['arr_{}'.format(args.seed-1)]
    print("test_node:")
    print(unknow_set)
    unknow_set = set(unknow_set)

    full_set = set(range(0, X.shape[0]))
    know_set = full_set - unknow_set
    training_set_s = training_set[:, list(know_set),:]  # get the training data in the sample time period
    A_s = A[:, list(know_set)][list(know_set), :]  # get the observed adjacent matrix from the full adjacent matrix,
    # the adjacent matrix are based on pairwise distance,
    # so we need not to construct it for each batch, we just use index to find the dynamic adjacent matrix
    return A, X, training_set, test_set, unknow_set, full_set, know_set, training_set_s, A_s, capacity,mean,std


"""
Define the test error
"""


def test_error(STmodel, unknow_set, test_data, A_s, Missing0, device):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """
    test_truth=test_data[:,:,0]  #tn
    unknow_set = set(unknow_set)
    # time_dim = STmodel.time_dimension
    time_dim=25
    test_omask = np.ones(test_data.shape)
    test_truth_omask=np.ones(test_truth.shape)  #tn
    if Missing0 == True:
        test_omask[test_data == 0] = 0
        test_truth_omask[test_truth == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')    #tn3
    test_truth=(test_truth*test_truth_omask).astype('float32')
    test_inputs_s = test_inputs

    missing_index = np.ones(np.shape(test_data))  # 0表未知点，1表示已知点 tn3
    missing_index[:, list(unknow_set),:] = 0    #缺失点的位置用0表示
    missing_index_s = missing_index
    missing_index_truth=missing_index[:,:,0]

    o = np.zeros([test_data.shape[0] // time_dim * time_dim,test_inputs_s.shape[1]])

    Ads=None
    for i in range(0, test_data.shape[0] // time_dim * time_dim, time_dim):
        inputs = test_inputs_s[i:i + time_dim, :]
        missing_inputs = missing_index_s[i:i + time_dim, :,:]
        T_inputs = inputs * missing_inputs
        T_inputs[:, :, 0:1] = (T_inputs[:, :, 0:1] - mean) / std
        # T_inputs = T_inputs / E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis=0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32')).to(device)  #bnt

        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32')).to(device)
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32')).to(device)

        imputation,Ads,Afs= STmodel(T_inputs, A_q, A_h)
        imputation = imputation.to(device).data.cpu().numpy()

        o[i:i + time_dim, :] = imputation[0, :, :]

    if dataset == 'NREL':
        o = o * capacities[None, :]
    else:
        # o = o * E_maxvalue
        o = o * std + mean

    truth = test_truth[0:test_data.shape[0] // time_dim * time_dim]
    o[missing_index_truth[0:test_data.shape[0] // time_dim * time_dim] == 1] = truth[
        missing_index_truth[0:test_data.shape[0] // time_dim * time_dim] == 1]  # 把真实值的已知点数据覆盖预测值的已知点数据，使其不影响误差计算

    test_mask = 1 - missing_index_truth[0:test_data.shape[0] // time_dim * time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0

    MAE = np.sum(np.abs(o - truth)) / np.sum(test_mask)


    RMSE = np.sqrt(np.sum((o - truth) * (o - truth)) / np.sum(test_mask))
    MAPE = np.sum(np.abs(o - truth) / (truth + 1e-5)) / np.sum(test_mask)

    return MAE, RMSE, MAPE, o, truth,Ads,Afs

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

if __name__ == "__main__":

    args = parse_arg()

    dataset = args.dataset
    n_o_n_m = args.n_s
    h = args.h
    z = args.z
    K = args.K
    n_m = args.n_m
    n_u = args.n_u
    max_iter = args.epochs
    learning_rate = args.learning_rate
    E_maxvalue = args.E_maxvalue
    batch_size = args.batch_size
    to_plot = True
    device = torch.device(f"cuda:{args.gpu_id}")
    seed = args.seed
    exp_id=args.id
    exp_group_id=args.gid
    patience=args.patience
    logging.basicConfig(filename=f'log/{dataset}/{exp_group_id}.log',level=logging.INFO)
    logging.info("#########################################################################################################")
    logging.info('[experiment_id={}]  seed={}  z={}  K={}  learning_rate={}  E={}  batch={} epoch={}'.format(exp_id,seed,z,K,learning_rate,E_maxvalue,batch_size,max_iter))
    logging.info(f'n_s={n_o_n_m}, n_m={n_m}, n_u={n_u}')
    logging.info(f'residual={args.residual},dilation={args.dilation},skip={args.skip},end={args.end},blocks={args.blocks},layers={args.layers}')
    logging.info(f'aux_h={args.aux_h},aux_layer={args.aux_layer}')
    seed_torch(seed)
    save_path = "./result_best/%s/%s" % (dataset,exp_id)
    best_model_path = f'model/{dataset}/best_model{exp_id}.pt'

    # load dataset
    A, X, training_set, test_set, unknow_set, full_set, know_set, training_set_s, A_s, capacity,mean,std= load_data(dataset)
    # Define model
    STmodel = gwnet(device=device,residual_channels=args.residual,dilation_channels=args.dilation,skip_channels=args.skip,end_channels=args.end,blocks=args.blocks,layers=args.layers,aux_h=args.aux_h,aux_layer=args.aux_layer) # The graph neural networks
    STmodel.to(device)
    # summary(STmodel, ( 25, 104, 3),4, device='cuda')
    criterion = nn.MSELoss()
    criterion2=nn.L1Loss()
    optimizer = optim.Adam(STmodel.parameters(), lr=learning_rate)
    RMSE_list = []
    MAE_list = []
    MAPE_list = []
    pred = []
    truth = []
    print('##################################    start training    ##################################')
    print(f"use gpu {args.gpu_id},  seed={seed},  experiment_id={exp_id}")
    best_mae = 100000
    early_stop_trigger=0
    for epoch in range(max_iter):
        time_s = time.time()
        # for i in range(1):
        for i in range(training_set.shape[0] // (h * batch_size)):
            t_random = np.random.randint(0, high=(training_set_s.shape[0] - h), size=batch_size, dtype='l')
            know_mask = set(random.sample(range(0, training_set_s.shape[1]), n_o_n_m))  # sample n_o + n_m nodes
            feed_batch = []
            for j in range(batch_size):
                # feed_batch.append(training_set_s[t_random[j]: t_random[j] + h, :][:, list(know_mask)]) #generate 8 time batches
                feed_batch.append(training_set_s[t_random[j]:t_random[j] + h, list(know_mask)])

            inputs = np.array(feed_batch)
            inputs_omask = np.ones(np.shape(inputs))    #b t n 3
            if not dataset == 'NREL':
                inputs_omask[
                    inputs == 0] = 0  # We found that there are irregular 0 values for METR-LA, so we treat those 0 values as missing data,
                inputs_omask[:,:,:,1:3]=1
                # For other datasets, it is not necessary to mask 0 values

            missing_index = np.ones((inputs.shape)) #btn3
            for j in range(batch_size):
                missing_mask = random.sample(range(0, n_o_n_m), n_m)  # Masked locations
                missing_index[j, :, missing_mask] = 0
            if dataset == 'NREL':
                Mf_inputs = inputs * inputs_omask * missing_index / capacities[:, None]
            else:
                Mf_inputs = inputs * inputs_omask * missing_index   # normalize the value according to experience
                Mf_inputs[..., 0] = (Mf_inputs[..., 0] - mean) / std
            Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32')).to(device)
            mask = torch.from_numpy(inputs_omask.astype('float32')).to(
                device)  # The reconstruction errors on irregular 0s are not used for training
            # print('Mf_inputs.shape = ',Mf_inputs.shape)

            A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]  # Obtain the dynamic adjacent matrix
            A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32')).to(device)
            A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32')).to(device)

            if dataset == 'NREL':
                outputs = torch.from_numpy(inputs / capacities[:, None]).to(device)
            else:
                outputs = (inputs[..., 0:1] - mean) / std
                outputs = torch.from_numpy(outputs).to(device)  # The label

            optimizer.zero_grad()
            #Flops,params = profile(STmodel,inputs=(Mf_inputs, A_q, A_h))
            #print('Flops: % .4fG'%(Flops / 1000000000))
            #print('Number of parameters: % .4fM' % (params / 1000000))
            X_res,Ad ,Af= STmodel(Mf_inputs, A_q, A_h)  # Obtain the reconstruction
            mask=mask[:,:,:,0]
            outputs=outputs[:,:,:,0].float()
            loss = criterion(X_res * mask, outputs * mask)+criterion2(X_res * mask, outputs * mask)
            loss.backward()
            optimizer.step()  # Errors backward
        if not dataset == 'NREL':
            MAE_t, RMSE_t, MAPE_t, pred, truth,Ads,Afs = test_error(STmodel, unknow_set, test_set, A, True, device)
        else:
            MAE_t, RMSE_t, MAPE_t, pred, truth,Ads,Afs = test_error(STmodel, unknow_set, test_set, A, False, device)
        time_e = time.time()
        RMSE_list.append(RMSE_t)
        MAE_list.append(MAE_t)
        MAPE_list.append(MAPE_t)
        print(epoch, MAE_t, RMSE_t, MAPE_t, 'time=', time_e - time_s)

        if MAE_t < best_mae:
            best_mae = MAE_t
            best_rmse = RMSE_t
            best_mape = MAPE_t
            best_epoch = epoch
            # best_model = copy.deepcopy(STmodel.state_dict())
            torch.save(STmodel.state_dict(), best_model_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savez(save_path + "result.npz", pred=pred, truth=truth,unknow_set=list(unknow_set),Ad=Ads.cpu().detach().numpy(),As = A,Af = Afs.cpu().detach().numpy())
            early_stop_trigger=0
        else:
            early_stop_trigger+=1
            if early_stop_trigger >= patience:
                print('early stop at epoch %d ' % (epoch))
                break

    logging.info(f'best test epoch={best_epoch}, test_mae={best_mae}, test_rmse={best_rmse}, tesy_mape={best_mape}')
    print("epoch = ", best_epoch, "     test_mae = ", best_mae, "    test_rmse = ", best_rmse, "    test_mape = ", best_mape)




