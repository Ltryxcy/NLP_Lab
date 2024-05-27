from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# 读取数据集函数
def read_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            yield line.strip()  # 去除首尾空白并返回单行文本


# 本地数据集文件路径
train_data_de_path = "./training/train.de"
train_data_en_path = "./training/train.en"
valid_data_de_path = "./validation/val.de"
valid_data_en_path = "./validation/val.en"

# 从本地文件中读取数据
train_de_sentences = list(read_dataset(train_data_de_path))
train_en_sentences = list(read_dataset(train_data_en_path))
valid_de_sentences = list(read_dataset(valid_data_de_path))
valid_en_sentences = list(read_dataset(valid_data_en_path))

# 合并德语和英语句子为成对数据
train_dataset = list(zip(train_de_sentences, train_en_sentences))
valid_dataset = list(zip(valid_de_sentences, valid_en_sentences))

# 创建分词器
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# 生成词表
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3  # 特殊token
UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM = '<unk>', '<pad>', '<bos>', '<eos>'  # 未知token、填充token、开始token、结束token

de_tokens = []  # 德语token列表
en_tokens = []  # 英语token列表
for de, en in train_dataset:
    de_tokens.append(de_tokenizer(de))
    en_tokens.append(en_tokenizer(en))

# 将列表转换为词表
de_vocab = build_vocab_from_iterator(
    de_tokens,
    specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM],
    special_first=True)  # 德语token词表
de_vocab.set_default_index(UNK_IDX)  # 设置未知token索引
en_vocab = build_vocab_from_iterator(
    en_tokens,
    specials=[UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM],
    special_first=True)  # 英语token词表
en_vocab.set_default_index(UNK_IDX)  # 设置未知token索引


# 句子特征预处理
def de_preprocess(de_sentence):
    tokens = de_tokenizer(de_sentence)  # 分词
    tokens = [BOS_SYM] + tokens + [EOS_SYM]  # 加上开始、结束token
    ids = de_vocab(tokens)  # 词汇表编码
    return tokens, ids  # 返回原始分词后的单词列表tokens以及对应的整数索引ids


def en_preprocess(en_sentence):
    tokens = en_tokenizer(en_sentence)  # 分词
    tokens = [BOS_SYM] + tokens + [EOS_SYM]  # 加上开始、结束token
    ids = en_vocab(tokens)  # 词汇表编码
    return tokens, ids  # 返回原始分词后的单词列表tokens以及对应的整数索引ids
