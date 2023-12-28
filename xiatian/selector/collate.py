import torch as t
from mectec import conf

def _pad_tensor(vec, max_len, dim, pad_value):    
    pad_size = list(vec.shape)
    pad_size[dim] = max_len - vec.size(dim)
    zeros = t.zeros(*pad_size, dtype=vec.dtype)
    return t.cat([vec, t.fill(zeros, pad_value)], dim=dim)


def pad_collate(batch):
    token_ids, type_ids, attention_mask = nolabel_pad(batch)
    # labels
    labels = t.tensor([x[-1] for x in batch]) # batch_size
    
    return token_ids, type_ids, attention_mask, labels


def nolabel_pad(batch):
    """输入对齐，batch中不带预测标记的情况"""
    dim = 0
    #max_len = max(map(lambda x: len(x[0]), batch))
    max_len = conf.MAX_INPUT_LEN
    
    # PAD_ID = 0
    token_ids = t.stack([_pad_tensor(x[0], max_len, dim, 0) for x in batch])
    # [SEP]分割的第3部分的type为0，填充的地方也采用0
    type_ids = t.stack([_pad_tensor(x[1], max_len, dim, 0) for x in batch])
    # 不需要attention的地方设置为0
    mask = t.stack([_pad_tensor(x[2], max_len, dim, 0) for x in batch])
    
    return token_ids, type_ids, mask