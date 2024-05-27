import torch

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 最长序列（受限于位置编码)
max_len = 5000
