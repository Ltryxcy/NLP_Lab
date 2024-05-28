from torch import nn
import torch
from dataset import en_preprocess, de_preprocess, train_dataset, en_vocab, de_vocab, PAD_IDX
from trans import Trans
from torch.utils.data import DataLoader, Dataset
from config import DEVICE, max_len
from torch.nn.utils.rnn import pad_sequence
import os
# import torch
import torch.nn as nn

from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

# 检查保存的模型的目录是否存在，如果不存在则创建
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')


# 数据集
class De2EnDataset(Dataset):

    def __init__(self):
        super().__init__()

        self.enc_x = []
        self.dec_x = []
        for de, en in train_dataset:
            # 分词
            de_tokens, de_ids = de_preprocess(de)
            en_tokens, en_ids = en_preprocess(en)
            # 跳过超出最大序列长度的部分
            if len(de_ids) > max_len or len(en_ids) > max_len:
                continue
            self.enc_x.append(de_ids)
            self.dec_x.append(en_ids)

    def __len__(self):
        return len(self.enc_x)  # 返回数据集中句子的数量

    def __getitem__(self, index):
        # 返回对应位置上的编码器输入和解码器输入的id序列
        return self.enc_x[index], self.dec_x[index]


# 在批量加载数据时对数据进行预处理
def collate_fn(batch):
    enc_x_batch = []
    dec_x_batch = []
    # 遍历batch中的每个样本,转换为张量并存储
    for enc_x, dec_x in batch:
        enc_x_batch.append(torch.tensor(enc_x, dtype=torch.long))
        dec_x_batch.append(torch.tensor(dec_x, dtype=torch.long))

    # batch内序列填充
    pad_enc_x = pad_sequence(enc_x_batch, True, PAD_IDX)
    pad_dec_x = pad_sequence(dec_x_batch, True, PAD_IDX)
    return pad_enc_x, pad_dec_x


if __name__ == '__main__':
    # de翻译en的数据集
    dataset = De2EnDataset()
    dataloader = DataLoader(dataset,
                            batch_size=250,
                            shuffle=True,
                            num_workers=4,
                            persistent_workers=True,
                            collate_fn=collate_fn)

    # 模型
    try:
        transformer = torch.load('checkpoints/model.pth')
    except:
        transformer = Trans(enc_vocab_size=len(de_vocab),
                            dec_vocab_size=len(en_vocab),
                            emb_size=512,
                            query_key_size=64,
                            value_size=64,
                            hidden_size=2048,
                            head_num=8,
                            block_num=6,
                            dropout=0.1,
                            max_len=max_len).to(DEVICE)  # 建立模型

    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)  # 序列的pad词不参与损失计算
    optimizer = torch.optim.SGD(transformer.parameters(), lr=1e-3, momentum=0.99)

    # 保存loss，用于绘图
    loss_list = []

    # 训练模型
    transformer.train()  # 训练模式
    EPOCHS = 1000
    for epoch in range(EPOCHS):
        # lr = pow(512, -0.5) * min((epoch+1e-9)**(-0.5),
        #                           (epoch+1e-9) * pow(4000, -1.5))  # 根据论文公式，计算学习率
        # optimizer = torch.optim.Adam(transformer.parameters(),
        #                              lr=lr,
        #                              betas=(0.9, 0.98),
        #                              eps=1e-9)  # 设置Adam优化器
        batch_i = 0  # batch计数
        loss_sum = 0
        print('epoch:{}'.format(epoch))
        for pad_enc_x, pad_dec_x in dataloader:
            real_dec_z = pad_dec_x[:, 1:].to(DEVICE)  # decoder正确输出
            pad_enc_x = pad_enc_x.to(DEVICE)
            pad_dec_x = pad_dec_x[:, :-1].to(DEVICE)  # decoder实际输入
            dec_z = transformer(pad_enc_x, pad_dec_x)  # decoder实际输出

            batch_i += 1
            loss = loss_fn(dec_z.view(-1, dec_z.size()[-1]), real_dec_z.view(-1))
            loss_sum += loss.item()  # 计算一个epoch内所有batch的loss之和
            print('epoch:{} batch:{} loss:{}'.format(epoch, batch_i, loss.item()))

            optimizer.zero_grad()
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
        torch.save(transformer, 'checkpoints/model_1.pth'.format(epoch))  # 保存模型
        loss_list.append(loss_sum / batch_i)  # 计算一个epoch内所有batch的loss均值

    # 画出loss曲线
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(title='Loss Curve', xlabel='epoch', ylabel='loss')
    ax.plot(range(len(loss_list)), loss_list)
    plt.savefig('loss.png')
    plt.show()
    print('done')
