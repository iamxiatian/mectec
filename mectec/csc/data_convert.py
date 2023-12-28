from typing import NamedTuple, List, Tuple
import torch
from mectec.util.lang_util import get_transform_edits
from mectec.conf import GEC_TOKEN_START, MAX_INPUT_LEN
from mectec.vocab import pinyin_vocab, megec_vocab
from mectec.csc.common import ErrorItem, OpEdit

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'

# 句子对齐，打上纠正标记和检测是否有错的标记，对应的结果
class AlignResult(NamedTuple):
    src_tokens: List[str]
    tgt_tokens: List[str]
    label_ids: List[str]
    dtag_ids: List[str]
    
SentenceError = List[ErrorItem]

def get_gec_errors(src_sentences:List[str], 
                   tgt_sentences:List[str]) -> List[SentenceError]:
    """
    由原始句子变换为目标句子时，对应的错误标记信息
    
    Args:
    src_sentences(List[str]): 原始句子序列
    tgt_sentences(List[str]): 目标句子序列
    
    输出形式，每个句子一个ErrorItem构成的列表，结构如下：
    [
        {'pos':原句中的位置, 
        'type':错误类型（别字/多字/少字）,
        'src':'原字符', 
        'tgt':目标字符}, 
        ...
    ]
    """
    error_sentences = []
    for src_sentence, tgt_sentence in zip(src_sentences, tgt_sentences):
        src_sentence = src_sentence.strip() \
                            .replace(' ', '')\
                            .replace("\t", "")\
                            .replace("	", "")
        diffs = get_transform_edits(src_sentence, tgt_sentence)
        errors = []
        for diff in diffs:
            tag, src_start, src_end, tgt_start, tgt_end = diff
            src_part = src_sentence[src_start:src_end]
            tgt_part = tgt_sentence[tgt_start:tgt_end]
            if tag == 'equal':
                continue
            elif tag == 'delete':
                errors.extend(
                    ErrorItem(j, '多字', src_part[j - src_start], '')
                            for j in range(src_start, src_end)
                )
            elif tag == 'insert':
                errors.append(ErrorItem(src_start,'少字', '', tgt_part))
            else:
                errors.extend(
                    ErrorItem(src_start + j, '别字', src_part[j], tgt_token)
                            for j, tgt_token in enumerate(tgt_part)
                )
        error_sentences.append(errors)
    return error_sentences


def convert_edits_to_ids(
        src_tokens, 
        edits:List[OpEdit]) -> Tuple[List[int], List[int]]:
    """
    根据编辑动作，转换形成纠正标签和检测标签的id列表，形成网络模型输出的预期结果数据。
    由于原始字符串的第一个位置可能是插入操作($A)，此时，$A需要插入到前面不存在的token上面。
    因此，返回的label和tag标记的数量，会多一个元素。
    例如：
    京是中国的首都。 => 北京是中国的首都。
    此时，缺失的“北”对应的纠正标记为"$A_北"，
    对应到[[START], 京, 是, 中, 国, 的, 首, 都, 。]的“[START]”上面
    """    
    num_labels = len(src_tokens) + 1
    if not edits:
        label_ids = [megec_vocab.convert_label_to_id("$K")] * num_labels
        dtag_ids = [megec_vocab.convert_dtag_to_id("CORRECT")] * num_labels
        return label_ids, dtag_ids 

    label_ids, dtag_ids = [], []
    for i in range(num_labels):
        if operations := [edit.operation for edit in edits
                if edit.start_pos == i - 1 and edit.end_pos == i]:
            label_id = megec_vocab.convert_label_to_id(operations[0], "$K")
            label_ids.append(label_id)
            if label_id == megec_vocab.KEEP_ID:
                dtag_ids.append(megec_vocab.convert_dtag_to_id("CORRECT"))
            else:
                dtag_ids.append(megec_vocab.convert_dtag_to_id("INCORRECT"))
        else:
            label_ids.append(megec_vocab.convert_label_to_id("$K"))
            dtag_ids.append(megec_vocab.convert_dtag_to_id("CORRECT"))
    return label_ids, dtag_ids


def make_bert_input(tokenizer, tokens, max_len):
    tokens = tokens[:max_len - 2] # 如果达到了最大长度，则空出最大长度的两个位置来，以便在前后补上[CLS]、[SEP]
    tokens = [CLS] + tokens + [SEP]
    seq_len = len(tokens) # 实际的token序列长度
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    if seq_len < max_len:
        # 有token的地方mask为1，否则为0
        mask = [1] * seq_len + [0] * (max_len - seq_len)
        token_ids += ([0] * (max_len - seq_len))
    else:
        mask = [1] * max_len
    segment_ids = [0] * max_len
    
    token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    mask = torch.as_tensor(mask, dtype=torch.long)
    segment_ids = torch.as_tensor(segment_ids, dtype=torch.long)
    
    return token_ids, mask, segment_ids
    
    
def make_train_input(tokenizer, tokens, label_ids, dtag_ids):
    """
    针对一个句子的词条序列，构造网络的输入数据，形成对应的attention_mask,segment_ids
    以及拼音和字形的输入信息。
    
    Returns:
        tokens: 一个句子对应的词条序列
        correct_label_ids: 纠正标记的id序列
        detect_tag_ids: 用于检测任务的标签id序列
        mask: attention mask
        segment_ids: segment id
        pinyin_ids: 拼音id序列
    """
    assert len(tokens) == len(label_ids) == len(dtag_ids) 
    # 空出最大长度的两个位置以便补上[CLS]、[SEP]
    tokens = tokens[:MAX_INPUT_LEN - 2] 
    label_ids = label_ids[:MAX_INPUT_LEN - 2]
    dtag_ids = dtag_ids[:MAX_INPUT_LEN - 2]

    #FIXME, 之前实验-1更好，why？
    default_dtag_id = megec_vocab.CORRECT_ID # 默认为无错误
    default_label_id = megec_vocab.KEEP_ID # 默认保持不变

    tokens = [CLS] + tokens + [SEP]
    label_ids = [default_label_id] + label_ids + [default_label_id]
    dtag_ids = [default_dtag_id] + dtag_ids + [default_dtag_id]

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    pinyin_ids = pinyin_vocab.convert_tokens_to_ids(tokens, padding=False)
    assert len(token_ids) == len(pinyin_ids)
    return token_ids, pinyin_ids, label_ids, dtag_ids


def make_predict_input(tokenizer, tokens):
    """
    针对一个句子的词条序列tokens，构造预测时网络的输入数据，形成对应的拼音和字形的输入信息。
    """
    tokens = [GEC_TOKEN_START] + tokens
    tokens = tokens[:MAX_INPUT_LEN - 2] 
    tokens = [CLS] + tokens + [SEP]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    pinyin_ids = pinyin_vocab.convert_tokens_to_ids(tokens, padding=False)
    return token_ids, pinyin_ids


def align_sequences(tokenizer, 
                    src_sentence:str,
                    tgt_sentence:str) -> AlignResult:
    """
    根据句子对(src_sentence, tgt_sentence)进行对齐， 
    生成纠错标签labels和差错标签tags, 原句子会补充上[START]标签
    """
    src_tokens:List[str] = tokenizer.tokenize(src_sentence)
    tgt_tokens:List[str] = tokenizer.tokenize(tgt_sentence)
    return align_tokens(src_tokens, tgt_tokens)
    
    
def align_tokens(src_tokens:list[str], tgt_tokens:list[str]) -> AlignResult:
    """
    根据句子对(src_tokens, tgt_tokens)进行对齐， 生成纠错标签labels和差错标签tags, 
    原句子会补充上[START]标签
    """
    diffs = get_transform_edits(src_tokens, tgt_tokens)
    edits = []
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        tgt_part = tgt_tokens[j1:j2]
        if tag == 'equal':
            continue
        elif tag == 'delete':
            for j in range(i1, i2):
                edit = OpEdit(j, j + 1, '$D')
                edits.append(edit)
        elif tag == 'insert':
            for tgt_token in tgt_part:
                edit = OpEdit(i1 - 1, i1, f"$A_{tgt_token}")
                edits.append(edit)
        else:
            for j, tgt_token in enumerate(tgt_part):
                edit = OpEdit(i1 + j, i1 + j + 1, f"$R_{tgt_token}")
                edits.append(edit)
    label_ids, dtag_ids = convert_edits_to_ids(src_tokens, edits)
    return AlignResult([GEC_TOKEN_START] + src_tokens, 
                       [GEC_TOKEN_START] + tgt_tokens, 
                       label_ids, 
                       dtag_ids)

