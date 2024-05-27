from torch import nn
from decoder import Decoder
from encoder import Encoder


class Trans(nn.Module):

    def __init__(self,
                 enc_vocab_size,
                 dec_vocab_size,
                 emb_size,
                 query_key_size,
                 value_size,
                 hidden_size,
                 head_num,
                 block_num,
                 dropout=0.1,
                 max_len=5000):
        super().__init__()
        self.encoder = Encoder(enc_vocab_size, emb_size, query_key_size, value_size,
                               hidden_size, head_num, block_num, dropout,
                               max_len)  # 编码器
        self.decoder = Decoder(dec_vocab_size, emb_size, query_key_size, value_size,
                               hidden_size, head_num, block_num, dropout,
                               max_len)  # 解码器

    def forward(self, encoder_x, decoder_x):
        encoder_z = self.encode(encoder_x)
        return self.decode(decoder_x, encoder_z, encoder_x)

    def encode(self, encoder_x):
        encoder_z = self.encoder(encoder_x)
        return encoder_z

    def decode(self, decoder_x, encoder_z, encoder_x):
        decoder_z = self.decoder(decoder_x, encoder_z, encoder_x)
        return decoder_z
