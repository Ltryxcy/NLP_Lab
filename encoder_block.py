from torch import nn
from multihead_attn import MultiHeadAttention

# 编码子层
class EncoderBlock(nn.Module):

    def __init__(self, emb_size, query_key_size, value_size, hidden_size, head_num):
        super().__init__()

        self.multihead_attn = MultiHeadAttention(emb_size, query_key_size, value_size,
                                                 head_num)  # 多头注意力
        self.z_linear = nn.Linear(head_num * value_size,
                                  emb_size)  # 线性层，将多头注意力模块的输出调整为与输入嵌入大小相同的尺寸。
        self.addnorm1 = nn.LayerNorm(emb_size)  # 层归一化，按last dim做norm

        # 前馈神经网络
        self.feedforward = nn.Sequential(nn.Linear(emb_size,
                                                   hidden_size), nn.ReLU(),
                                         nn.Linear(hidden_size, emb_size))
        self.addnorm2 = nn.LayerNorm(emb_size)  # 层归一化，按last dim做norm

    def forward(self, x, attn_mask):  # x: (batch_size,seq_len,emb_size)
        z = self.multihead_attn(x, x, attn_mask)  # 多头注意力
        z = self.z_linear(z)  # 线性层，将z维度变成和x相同
        output1 = self.addnorm1(z + x)  # 层归一化

        z = self.feedforward(output1)  # 前馈神经网络
        return self.addnorm2(z + output1)  # 层归一化
