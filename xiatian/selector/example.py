from dataclasses import dataclass
from copy import copy
from dataclasses import dataclass

import torch as t
from mectec.util.lang_util import get_transform_edits
from mectec import conf
from .collate import nolabel_pad

@dataclass
class Example:
    tokens: list[str] # 句子对应的token信息，报错位置用WHOAMI替代
    ref_token1: str # 第一个参考token
    ref_token2: str #第二个参考token
    
MASK = conf.BERT_TOKEN_MASK

def generate_examples(src_tokens:list[str], 
                      tgt_tokens:list[str]) -> list[Example]:
    """
    根据句子对(src_tokens, tgt_tokens)进行对齐， 找出错误信息，生成纠错案例。
    一个句子对会生成0到多个样本，每一个字符错误生成一个，返回三元组，三元组格式为：
    （把错误位置用MASK遮盖，报错位置用WHOAMI替代的tokens序列，原始的token，目标token）
    """
    
    diffs = get_transform_edits(src_tokens, tgt_tokens)

    # 先把所有有错误位置的地方改成MASK，需要在源串中“insert”位置的地方，不加MASK
    offset = 0
    masked_tokens = copy(src_tokens)
    # 挑选出有错误的位置信息
    masked_location = filter(
        lambda op: op[0] in ['replace', 'delete', 'insert'],
        diffs)
    for diff in masked_location:
        tag, i1, i2, j1, j2 = diff
        if tag == 'insert':
            # 对于insert，也在masked_tokens中增加了一个[MASK]，因此改变了长度，
            # 后面需要用offset记录增加的数量
            masked_tokens = masked_tokens[:i1] + [MASK] + masked_tokens[i2:]  
            offset += 1  
        else:
            for i in range(i1, i2):
                masked_tokens[i + offset] = MASK

    # examples中保存了三元组信息：（把错误位置用MASK遮盖，报错位置用WHOAMI替代的tokens序列，原始的token，目标token）
    examples = []

    offset = 0
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        tgt_part = tgt_tokens[j1:j2]
        if tag == 'delete':
            for i in range(i1, i2):
                tokens = copy(masked_tokens)
                tokens[i + offset] = conf.GEC_TOKEN_WHOAMI
                examples.append(
                    Example(tokens, src_tokens[i], conf.GEC_TOKEN_DEL))
        elif tag == 'equal':
            continue
        elif tag == 'insert':
            # 只保留一个
            tokens = copy(masked_tokens)
            tokens[i1 + offset] = conf.GEC_TOKEN_WHOAMI
            #tokens = tokens[:i1] + [conf.GEC_TOKEN_WHOAMI] + tokens[i2:]
            examples.append(Example(tokens, conf.GEC_TOKEN_DEL, tgt_part[0]))
            offset += 1
        else:
            for j, tgt_token in enumerate(tgt_part):
                tokens = copy(masked_tokens)
                tokens[i1+j + offset] = conf.GEC_TOKEN_WHOAMI
                examples.append(Example(tokens, src_tokens[i1+j], tgt_token))
    return examples


def make_input(example:Example, is_error_first:bool=True):
    """将一个样本转换为网络的输入Tensor"""
    tokens = [conf.BERT_TOKEN_CLS] + example.tokens + [conf.BERT_TOKEN_SEP]
    if is_error_first:
        tokens = tokens + [
                example.ref_token1, conf.BERT_TOKEN_SEP, 
                example.ref_token2,conf.BERT_TOKEN_SEP
            ]
    else:
        tokens = tokens + [
                example.ref_token2, conf.BERT_TOKEN_SEP, 
                example.ref_token1,conf.BERT_TOKEN_SEP
            ]
    
    token_ids = conf.bert_tokenizer.convert_tokens_to_ids(tokens)
    type_ids = [0] *(2+ len(example.tokens)) + [1, 1, 0, 0]  
    attention_mask = [1] * len(tokens)
    return t.tensor(token_ids), t.tensor(type_ids), t.tensor(attention_mask)

def make_inputs(examples:list[Example])->tuple[t.Tensor, t.Tensor, t.Tensor]:
    batch = [make_input(e) for e in examples]
    return nolabel_pad(batch)