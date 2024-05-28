from torch import nn
import torch
from emb import EmbeddingWithPosition
from dataset import PAD_IDX
from decoder_block import DecoderBlock
from config import DEVICE


# 解码器
class Decoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 emb_size,
                 query_key_size,
                 value_size,
                 hidden_size,
                 head_num,
                 block_num,
                 dropout=0.1,
                 max_len=5000):
        super().__init__()
        # 嵌入层,将词汇表中的单词映射到向量，并位置编码
        self.emb = EmbeddingWithPosition(vocab_size, emb_size, dropout, max_len)
        # nblocks个解码器子层
        self.decoder_blocks = nn.ModuleList()
        for _ in range(block_num):
            self.decoder_blocks.append(
                DecoderBlock(emb_size, query_key_size, value_size, hidden_size, head_num))

        # 输出向量词概率Logits
        self.linear = nn.Linear(emb_size, vocab_size)

    def forward(self, x, encoder_z, encoder_x):
        # 目标序列的pad掩码,用于屏蔽掉目标序列中的padding位置
        first_attn_mask = (x == PAD_IDX).unsqueeze(1).expand(x.size()[0], x.size()[1], x.size()[1]).to(DEVICE)
        # 目标序列的向后看掩码，用于屏蔽掉位置向后的依赖关系，只允许位置向前的依赖关系。
        first_attn_mask = first_attn_mask | torch.triu(torch.ones(x.size()[1], x.size()[1]), diagonal=1).bool().unsqueeze(0).expand(x.size()[0], -1, -1).to(DEVICE)

        # 根据来源序列的pad掩码，遮盖decoder对其pad部分的注意力
        # 找出encoder_x中的PAD_IDX位置，然后扩展此掩码以匹配解码器的目标序列长度
        second_attn_mask = (encoder_x == PAD_IDX).unsqueeze(1).expand(encoder_x.size()[0], x.size()[1], encoder_x.size()[1]).to(DEVICE)

        x = self.emb(x)  # 词嵌入
        for block in self.decoder_blocks:
            # 经过每个解码子层
            x = block(x, encoder_z, first_attn_mask, second_attn_mask)

        # 将解码器的隐藏表示映射到词汇表大小的向量，每个位置的向量代表该位置预测各个词的概率分布。
        return self.linear(x)
