import torch
from mectec.conf import MAX_INPUT_LEN
def pad_tensor(vec, max_len, pad_value=0):
    """
    args:
        vec - value list to pad
        max_len - the size to pad to
        pad_value - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = max_len - len(vec)
    vec += [pad_value] * pad_size 
    return torch.tensor(vec, dtype=torch.long)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, has_target):
        """
        args:
            has_target - bool, 如果为true，表示包含了预测目标，否则，
                不返回batch_correct_label_ids和batch_detect_tag_ids
        """
        self.has_target = has_target

    def train_collate(self, batch):
        """
        args:
            batch - list of list, see make_train_input in data_convert
            
        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        # MAX_INPUT_LEN = max(map(lambda x: len(x[0]), batch))
        batch_token_ids, batch_pinyin_ids = [], []
        batch_label_ids, batch_dtag_ids = [], []
        batch_masks = []
        batch_type_ids = []
        batch_seq_len = []
        for token_ids, pinyin_ids, correct_label_ids, detect_tag_ids in batch:
            token_num = len(token_ids)
            batch_token_ids.append(
                pad_tensor(token_ids, max_len=MAX_INPUT_LEN))
            
            batch_pinyin_ids.append(
                pad_tensor(pinyin_ids, max_len=MAX_INPUT_LEN))
            
            mask = torch.tensor([1] * token_num \
                + [0] * (MAX_INPUT_LEN - token_num), dtype=torch.long)
            
            batch_masks.append(mask)
            
            batch_type_ids.append(torch.zeros(MAX_INPUT_LEN, dtype=torch.long))
            
            # seq_len: 实际的token序列长度，不包含额外加上的[CLS]和[SEP]
            batch_seq_len.append(token_num - 2)
            
            batch_label_ids.append(
                pad_tensor(correct_label_ids, max_len=MAX_INPUT_LEN))
            
            batch_dtag_ids.append(
                pad_tensor(detect_tag_ids, max_len=MAX_INPUT_LEN))
            
        # stack all
        batch_token_ids = torch.stack(batch_token_ids, dim=0)
        batch_pinyin_ids = torch.stack(batch_pinyin_ids, dim=0)
        batch_masks = torch.stack(batch_masks, dim=0)
        batch_type_ids = torch.stack(batch_type_ids, dim=0)
        batch_seq_len = torch.tensor(batch_seq_len, dtype=torch.long)
        batch_label_ids = torch.stack(batch_label_ids, dim=0)
        batch_dtag_ids = torch.stack(batch_dtag_ids, dim=0)
    
        return batch_token_ids, batch_pinyin_ids, \
            batch_label_ids, batch_dtag_ids, \
            batch_masks, batch_type_ids, batch_seq_len
            

    def predict_collate(self, batch):
        """预测时的批数据处理"""
        # MAX_INPUT_LEN = max(map(lambda x: len(x[0]), batch))
        
        batch_token_ids, batch_pinyin_ids = [], []
        batch_masks = []
        batch_type_ids = []
        batch_seq_len = []
        for token_ids, pinyin_ids in batch:
            token_num = len(token_ids)
            batch_token_ids.append(
                pad_tensor(token_ids, max_len=MAX_INPUT_LEN))
            batch_pinyin_ids.append(
                pad_tensor(pinyin_ids, max_len=MAX_INPUT_LEN))
            mask = torch.tensor([1] * token_num \
                + [0] * (MAX_INPUT_LEN - token_num), dtype=torch.long)
            batch_masks.append(mask)
            batch_type_ids.append(torch.zeros(MAX_INPUT_LEN, dtype=torch.long))
            # seq_len: 实际的token序列长度，不包含额外加上的[CLS]和[SEP]
            batch_seq_len.append(token_num - 2)
            
        # stack all
        batch_token_ids = torch.stack(batch_token_ids, dim=0)
        batch_pinyin_ids = torch.stack(batch_pinyin_ids, dim=0)
        batch_masks = torch.stack(batch_masks, dim=0)
        batch_type_ids = torch.stack(batch_type_ids, dim=0)
        batch_seq_len = torch.tensor(batch_seq_len, dtype=torch.long)
        
        return batch_token_ids, batch_pinyin_ids, \
            batch_masks, batch_type_ids, batch_seq_len
                
    def __call__(self, batch):
        if self.has_target:
            return self.train_collate(batch) 
        else:
            return self.predict_collate(batch)