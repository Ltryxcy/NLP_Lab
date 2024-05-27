from torch import nn
import torch
import math


# 将词嵌入和固定位置编码结合
class EmbeddingWithPosition(nn.Module):

    def __init__(self, vocab_size, emb_size, dropout=0.1, max_len=5000):
        super().__init__()

        # 词嵌入。将序列中的每个词转换成emb向量, 其他形状不变
        self.seq_emb = nn.Embedding(vocab_size, emb_size)  # 调用nn.Embedding

        # 固定位置编码,创建位置编码矩阵。序列中每个位置有一个位置向量，也是emb_size维的
        position_idx = torch.arange(
            0, max_len, dtype=torch.float).unsqueeze(
                -1)  # 生成一个从0到seq_max_len-1的浮点数序列，然后将其展平为一维。

        # 按照Transformer论文中的公式计算正弦和余弦值，生成位置编码
        position_emb_fill = position_idx * torch.exp(
            -torch.arange(0, emb_size, 2) * math.log(10000.0) / emb_size)
        pos_encoding = torch.zeros(max_len, emb_size)
        pos_encoding[:, 0::2] = torch.sin(position_emb_fill)
        pos_encoding[:, 1::2] = torch.cos(position_emb_fill)
        self.register_buffer('pos_encoding', pos_encoding)  # 固定参数,不需要train

        # 添加dropout防过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.seq_emb(x)  # 词嵌入
        pos_encode = self.pos_encoding.unsqueeze(
            0)[:, :x.size()[1], :]  # 将位置编码张量扩展后截取,保证其长度与实际存在的序列长度匹配
        x = x + pos_encode  # 将位置编码向量与词嵌向量相加，合并两种信息。
        return self.dropout(x)  # dropout
