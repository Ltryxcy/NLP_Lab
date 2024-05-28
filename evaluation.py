import torchtext

torchtext.disable_torchtext_deprecation_warning()
import nltk
import torch
from dataset import de_preprocess, BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX, en_vocab
from config import DEVICE, max_len
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dataset import valid_dataset


# de翻译到en
def translate(transformer, de_sentence):
    # De分词
    de_tokens, de_ids = de_preprocess(de_sentence)  # 预处理句子
    if len(de_tokens) > max_len:  # De句子长度超过SEQ_MAX_LEN
        raise Exception('不支持超过{}的句子'.format(max_len))

    # Encoder阶段
    enc_x_batch = torch.tensor([de_ids], dtype=torch.long).to(DEVICE)  # 准备encoder输入
    encoder_z = transformer.encode(enc_x_batch)  # encoder编码

    # Decoder阶段
    en_token_ids = [BOS_IDX]  # 翻译结果
    while len(en_token_ids) < max_len:
        dec_x_batch = torch.tensor([en_token_ids], dtype=torch.long).to(DEVICE)  # 准备decoder输入
        decoder_z = transformer.decode(dec_x_batch, encoder_z, enc_x_batch)  # decoder解碼
        next_token_probs = decoder_z[0, dec_x_batch.size(-1) - 1, :]  # 序列下一个词的概率
        next_token_id = torch.argmax(next_token_probs)  # 下一个词ID
        en_token_ids.append(next_token_id)

        if next_token_id == EOS_IDX:  # 结束符
            break

    # 生成翻译结果
    en_token_ids = [
        id for id in en_token_ids
        if id not in [BOS_IDX, EOS_IDX, UNK_IDX, PAD_IDX]
    ]  # 忽略特殊字符
    en_tokens = en_vocab.lookup_tokens(en_token_ids)  # 词id序列转token序列
    return ' '.join(en_tokens)


# 导入相关库
from torchtext.data.metrics import bleu_score

if __name__ == '__main__':
    # 加载Adam模型
    transformer = torch.load('checkpoints/model_1.pth')
    transformer.eval()

    # 初始化 BLEU 分数计算
    bleu_scores1 = []
    bleu_scores2 = []

    # 遍历测试数据集
    for de, en in valid_dataset:
        # 获取模型翻译结果
        en_pred = translate(transformer, de)

        # 将模型翻译结果和参考翻译进行分词
        en_pred_tokens = en_pred.split()
        en_ref_tokens = en.split()

        # 计算句子级 BLEU 分数
        bleu_score_sentence = bleu_score([en_pred_tokens], [[en_ref_tokens]])
        bleu_scores1.append(bleu_score_sentence)

    # 计算平均 BLEU 分数
    avg_bleu_score = sum(bleu_scores1) / len(bleu_scores1)
    print(f'Average BLEU Score for Adam: {avg_bleu_score *100.0:.4f}')

    # 加载SGD模型
    transformer = torch.load('checkpoints/model.pth')
    transformer.eval()
    for de, en in valid_dataset:
        en_pred = translate(transformer, de)
        en_pred_tokens = en_pred.split()
        en_ref_tokens = en.split()

        bleu_score_sentence = bleu_score([en_pred_tokens], [[en_ref_tokens]])
        bleu_scores2.append(bleu_score_sentence)

    avg_bleu_score_2 = sum(bleu_scores2) / len(bleu_scores2)
    print(f'Average BLEU Score for SGD: {avg_bleu_score_2 *100.0:.4f}')
