from torch import nn
import torch
import math


# 多头注意力
class MultiHeadAttention(nn.Module):

    def __init__(self,
                 emb_size,
                 query_key_size,
                 value_size,
                 head_num):
        super().__init__()
        self.emb_size = emb_size  # 词嵌入维度
        self.query_key_size = query_key_size  # query和key的向量维度
        self.value_size = value_size  # value的向量维度
        self.head_num = head_num  # 多头注意力中的head数

        # 线性层,将输入的词嵌入向量转换为多头注意力的query向量
        self.w_q = nn.Linear(emb_size, head_num * query_key_size)
        # 线性层,将输入的词嵌入向量转换为多头注意力的key向量
        self.w_k = nn.Linear(emb_size, head_num * query_key_size)
        # 线性层,将输入的词嵌入向量转换为多头注意力的value向量
        self.w_v = nn.Linear(emb_size, head_num * value_size)

    def forward(self, x_q, x_k_v, attn_mask):
        q = self.w_q(x_q)  # 将输入的词嵌入向量转换为多头注意力的query向量
        k = self.w_k(x_k_v)  # 将输入的词嵌入向量转换为多头注意力的key向量

        # 多头兼容。将两个张量重塑转置，以适应多头注意力的计算
        q = q.view(q.size()[0], q.size()[1], self.head_num, self.query_key_size).transpose(1, 2)
        k = k.view(k.size()[0], k.size()[1], self.head_num, self.query_key_size).transpose(1, 2).transpose(2, 3)

        # 注意力矩阵
        # 使用矩阵乘法计算查询和键之间的点积，然后除以sqrt(self.query_key_size)来缩放注意力得分，防止数值过大。
        attn = torch.matmul(q, k) / math.sqrt(self.query_key_size)

        # 注意力分值处理
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.head_num, -1, -1)  # 将attn_mask扩展以适应多头注意力
        attn = attn.masked_fill(attn_mask, -1e9)  # 用-1e9填充掩码位置，这样在softmax操作中这些位置的得分将接近于0。
        attn = torch.softmax(attn, dim=-1)  # softmax操作，将注意力矩阵转换为概率分布

        # 注意力与V相乘
        v = self.w_v(x_k_v)  # 将key向量做线性变换
        v = v.view(v.size()[0], v.size()[1], self.head_num, self.value_size).transpose(1, 2)  # 将key向量做重塑和转置
        z = torch.matmul(attn, v)  # 使用注意力权重矩阵对key向量做加权求和
        # 对z转置并重塑
        z = z.transpose(1, 2)
        return z.reshape(z.size()[0], z.size()[1], -1)
