import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from utils import scaled_Laplacian, cheb_polynomial
import os

class GraphAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k ):
        super(GraphAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x):
        '''
        param x: [batch_size,len_q, nodes_number, d_model]
        return [batch_size, len_q, nodes_number, nodes_number]
        '''
        Q = self.W_Q(x)
        K = self.W_K(x).transpose(3,2)
        scores = torch.matmul(Q, K) / np.sqrt(self.d_k) #[batch_size, len_q, nodes_number, nodes_number]
        attn = F.softmax(scores, dim=3) 
        return attn

class Embedding(nn.Module):
    def __init__(self, nb_seq, d_Em, num_of_features):
        super(Embedding, self).__init__()
        self.nb_seq = nb_seq
        self.num_of_features = num_of_features
        self.pos_embed = nn.Embedding(nb_seq, d_Em)
        self.norm = nn.LayerNorm(d_Em)

    def forward(self, x, batch_size):
        '''
        param x (b,N,T,C)
        '''
        pos = torch.arange(self.nb_seq, dtype=torch.long).cuda()
        pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_of_features,
                                                   self.nb_seq)  
        x=x.reshape(batch_size,self.num_of_features,self.nb_seq,-1)
        embedding = x + self.pos_embed(pos)
        Emx = self.norm(embedding)
        return Emx

class SGA_Module(nn.Module):
    """ Spatial guided attention module"""
    def __init__(self, in_dim):
        super(SGA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        """
        Parameters: x (B,C,N,T)
        """
        B,C,N,T = x.size()
        proj_query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, N)
        energy = torch.bmm(proj_query, proj_key)  #B N N
        attention = self.softmax(energy)
        print(attention.size())
        matrix_slice = attention[:, 120:140, 120:140].mean(dim=0).detach().cpu().numpy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_slice, annot=False, fmt=".2f", cmap='viridis')
        plt.xlabel('Position')
        plt.ylabel('Position')
        plt.show()
        proj_value = self.value_conv(x).view(B, -1, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, N, T)
        out = self.gamma * out + x
        return out

class TGA_Module(nn.Module):
    """ Temporal guided attention module"""
    def __init__(self, in_dim):
        super(TGA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        """
        Parameters: x (B,C,T,N)
        """
        B,C,T,N = x.size()
        proj_query = self.query_conv(x).view(B, -1, T).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, T)
        energy = torch.bmm(proj_query, proj_key)  # B T T
        attention = self.softmax(energy)
        print(attention.size())
        matrix_slice = attention.mean(dim=0).detach().cpu().numpy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_slice, annot=False, fmt=".2f", cmap='viridis')
        plt.xlabel('Time Step')
        plt.ylabel('Time Step')
        plt.show()
        proj_value = self.value_conv(x).view(B, -1, T)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, T, N)
        out = self.gamma * out + x
        return out

class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials_spatial, cheb_polynomials_sem, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials_spatial = cheb_polynomials_spatial
        self.cheb_polynomials_sem = cheb_polynomials_sem
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials_spatial[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        self.spa_atten = GraphAttention(self.DEVICE, in_channels, out_channels)
        self.sem_atten =GraphAttention(self.DEVICE, in_channels, out_channels)


    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []
        spatial_atten = self.spa_atten(x.permute(0,3,1,2))
        sem_atten = self.sem_atten(x.permute(0,3,1,2))
        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k_spatial = self.cheb_polynomials_spatial[k]  # (N,N)
                T_k_sem = self.cheb_polynomials_sem[k]  # (N,N)
                T_k_with_at_spatial = T_k_spatial.mul(spatial_atten[:,time_step,:,:])   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化
                T_k_with_at_sem = T_k_sem.mul(sem_atten[:,time_step,:,:])   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化
                T_k_with_at = T_k_with_at_spatial + T_k_with_at_sem
                # T_k_with_at =T_k_with_at_sem
                theta_k = self.Theta[k]  # (in_channel, out_channel)
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)

class GTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides),padding=(0,1))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu

class TimeConv(nn.Module):
    def __init__(self,DEVICE,in_channels,out_channels,num_of_vertices):
        super(TimeConv,self).__init__()
        self.gtu=GTU(out_channels,1,3)
        self.slice_gtu=nn.ModuleList([GTU(out_channels,1,3) for _ in range(3)])
        self.W1=nn.Parameter(torch.FloatTensor(out_channels,num_of_vertices,12).to(DEVICE))
        self.W2=nn.Parameter(torch.FloatTensor(out_channels,num_of_vertices,12).to(DEVICE))
    def forward(self,x):
        "x:(B,N,C,T)"
        x=x.permute(0,2,1,3)
        x_slice=[]
        for i in range(3):
            x_slice.append(self.slice_gtu[i](x[:,:,:,i*4:(i+1)*4]))
        x_slice=torch.cat(x_slice,dim=3)
        x_glob=self.gtu(x)
        #output=torch.mul(self.W2,x_glob)
        output=torch.mul(self.W1,x_slice)+torch.mul(self.W2,x_glob)
        return  output.permute(0,2,1,3)

class DMSTCN_block(nn.Module):
    def __init__(self,DEVICE,in_channels,out_channels,cheb_polynomials_spatial,cheb_polynomials_sem,  num_of_vertices,K):
        super(DMSTCN_block,self).__init__()
        self.TimeBlock=TimeConv(DEVICE,in_channels,out_channels,num_of_vertices)
        self.In=nn.LayerNorm(out_channels)
        self.cheb_conv_SAt = cheb_conv_withSAt(K,cheb_polynomials_spatial,cheb_polynomials_sem, in_channels, out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
        
    def forward(self,x):
        "x:(B,T,N,C)" 
        "output: (B,T,N,C)"
        x_residual = self.residual_conv(x.permute(0, 3, 1, 2)).permute(0,2,3,1)
        x=x.permute(0,2,3,1)
        output=self.cheb_conv_SAt(x)
        output=self.TimeBlock(output).permute(0,3,1,2)
        output=self.In(F.relu(output+x_residual,inplace=False))
        return output

class DMSTGNN_submodule(nn.Module):
  def __init__(self, DEVICE, nb_block, in_channels, nb_filter, num_for_predict, len_input, num_of_vertices,K, cheb_polynomials_spatial,   cheb_polynomials_sem):
    super(DMSTGNN_submodule,self).__init__()
    
    self.num_for_predict = num_for_predict
    
    self.BlockList = nn.ModuleList([DMSTCN_block(DEVICE,in_channels,nb_filter, cheb_polynomials_spatial,   cheb_polynomials_sem, num_of_vertices,K)])

    self.BlockList.extend([DMSTCN_block(DEVICE,nb_filter,nb_filter, cheb_polynomials_spatial,   cheb_polynomials_sem, num_of_vertices,K) for _ in range(nb_block-1)])
    
    self.fc=nn.Conv2d(len_input,num_for_predict,kernel_size=(1,nb_filter*nb_block))

    self.len_input=len_input

    self.atten =GraphAttention(DEVICE, in_channels, nb_filter)

    self.nb_filter = nb_filter

    self.cheb_polynomials_spatial = cheb_polynomials_spatial

    self.cheb_polynomials_sem = cheb_polynomials_sem

    self.EmbedT = Embedding(len_input, num_of_vertices, nb_filter)

    self.EmbedS = Embedding(num_of_vertices, len_input, nb_filter)

    self.sga=SGA_Module(nb_filter)

    self.tga=TGA_Module(nb_filter)

    self.DEVICE = DEVICE
    
    self.to(DEVICE)

    self.sga_matrix = None  # 初始化 sga_matrix 属性
    self.tga_matrix = None  # 初始化 tga_matrix 属性
    
  def forward(self, x):
    '''
    :param x: (batch_size, len_input, N,  C)
    :return: (batch_size, num_for_predict, N, C)
    '''
    batch_size, len_time,num_of_vertices,in_channels=x.size()
    output = []
    for block in self.BlockList:
      x = block(x) 
      T_emb = self.EmbedT(x,batch_size)
      S_emb = self.EmbedS(x,batch_size)
      T_out = self.tga(T_emb).reshape(batch_size,len_time,num_of_vertices,self.nb_filter)
      S_out = self.sga(S_emb).reshape(batch_size,len_time,num_of_vertices,self.nb_filter)
      out = T_out + S_out
      output.append(out)

      #保存注意力矩阵
      self.sga_matrix = S_out.detach().cpu().numpy()
      self.tga_matrix = T_out.detach().cpu().numpy()

    output = torch.cat(output, dim=-1)
    output = self.fc(output).reshape(batch_size,len_time,num_of_vertices,in_channels)
    return output
    
def make_model(DEVICE, nb_block, in_channels, nb_filter, num_for_predict, len_input, num_of_vertices,K,adj_spatial,adj_sem):

    L_tilde_spatial = scaled_Laplacian(adj_spatial)
    cheb_polynomials_spatial = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde_spatial, K)]
    L_tilde_sem = scaled_Laplacian(adj_sem)
    cheb_polynomials_sem = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde_sem, K)]
    model = DMSTGNN_submodule(DEVICE, nb_block, in_channels, nb_filter, num_for_predict, len_input, num_of_vertices,K, cheb_polynomials_spatial,   cheb_polynomials_sem)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model 