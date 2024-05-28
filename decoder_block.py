from torch import nn
from multihead_attn import MultiHeadAttention


# 编码器子层
class DecoderBlock(nn.Module):

    def __init__(self,
                 emb_size,
                 query_key_size,
                 value_size,
                 hidden_size,
                 head_num):
        super().__init__()

        # 第1个多头注意力
        self.first_multihead_attn = MultiHeadAttention(emb_size, query_key_size, value_size, head_num)
        # 将多头注意力的输出转换会原始的嵌入维度
        self.z_linear1 = nn.Linear(head_num * value_size, emb_size)
        # 层归一化,有助于稳定训练过程并加速收敛
        self.addnorm1 = nn.LayerNorm(emb_size)

        # 第2个多头注意力
        self.second_multihead_attn = MultiHeadAttention(emb_size, query_key_size, value_size, head_num)
        self.z_linear2 = nn.Linear(head_num * value_size, emb_size)
        self.addnorm2 = nn.LayerNorm(emb_size)

        # 前馈网络。包含两个线性层，中间使用ReLU激活函数
        self.feedforward = nn.Sequential(nn.Linear(emb_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, emb_size))
        self.addnorm3 = nn.LayerNorm(emb_size)

    def forward(self, x, encoder_z, first_attn_mask, second_attn_mask):
        # 第1个多头注意
        z = self.first_multihead_attn(x, x, first_attn_mask)
        z = self.z_linear1(z)
        output1 = self.addnorm1(z + x)

        # 第2个多头注意力
        # second_attn_mask用于遮盖encoder序列的pad部分,避免decoder Q到它们
        z = self.second_multihead_attn(output1, encoder_z, second_attn_mask)
        z = self.z_linear2(z)
        output2 = self.addnorm2(z + output1)
        # 前馈网络
        z = self.feedforward(output2)
        return self.addnorm3(z + output2)
