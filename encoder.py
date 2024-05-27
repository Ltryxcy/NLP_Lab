from torch import nn
from encoder_block import EncoderBlock
from emb import EmbeddingWithPosition
from dataset import PAD_IDX
from config import DEVICE


class Encoder(nn.Module):

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
        self.emb = EmbeddingWithPosition(vocab_size, emb_size, dropout,
                                         max_len)  # 词嵌入并位置编码
        # nblocks个编码子层
        self.encoder_blocks = nn.ModuleList()
        for _ in range(block_num):
            self.encoder_blocks.append(
                EncoderBlock(emb_size, query_key_size, value_size, hidden_size, head_num))

    def forward(self, x):
        pad_mask = (x == PAD_IDX).unsqueeze(1)  # 增加维度，使掩码形状与后续操作兼容
        pad_mask = pad_mask.expand(x.size()[0],
                                   x.size()[1],
                                   x.size()[1])  # 将掩码沿特定维度复制，形成三维的mask

        pad_mask = pad_mask.to(DEVICE)

        x = self.emb(x)  # 词嵌入
        for block in self.encoder_blocks:
            x = block(x, pad_mask)  # 编码子层
        return x
