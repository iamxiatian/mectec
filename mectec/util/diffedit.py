from typing import List

punc_set = set('，。！：“”‘’？《》「」（）『』〈〉')
    
def diff_puncs(source:str, target:str) -> List[str]:
    """
    获取两个句子的标点符号差异，在原句子上打上标记，标记结果为：KEEP,DELETE, REPLACE_X
    >>> diff_puncs('我爱北，京天安门天安门上红旗。飞', '：我爱北京天安门，天安门上红：旗飞。')
    ['$A_：', '$K', '$K', '$K', '$D', '$K', '$K', '$K', '$A_，', '$K', '$K', '$K', '$K', '$A_：', '$K', '$D', '$A_。']
    """
    src_chars = ['[CLS]'] + list(source)
    tgt_chars = ['[CLS]'] + list(target)
    return diff_token_puncs(src_chars, tgt_chars)


def diff_token_puncs(src_tokens:List[str], tgt_tokens:List[str]) -> List[str]:
    src_idx, tgt_idx = 0, 0
    
    edits = []
    while src_idx < len(src_tokens) and tgt_idx < len(tgt_tokens):
        src_token = src_tokens[src_idx]
        tgt_token = tgt_tokens[tgt_idx]
        
        # 相等则KEEP
        if src_token == tgt_token:
            edits.append('$K')
            src_idx += 1
            tgt_idx +=1
            continue
        
        # 两个都是标点，则替换
        if src_token in punc_set and tgt_token in punc_set:
            edits.append('$R' + tgt_token)
            src_idx += 1
            tgt_idx +=1
            continue
        
        # src标点，tgt为非标点，则删除，如：我爱北，京 => 我爱北京
        if src_token in punc_set and tgt_token not in punc_set:
            edits.append('$D')
            src_idx += 1
            continue
        
        # src非标点，tgt为标点，则在前一个上标记$A_X，如：我爱北京 => 我爱北，京
        # 则将“北”的标签由“$K”修改为 "$A_,"
        if src_token not in punc_set and tgt_token in punc_set:
            edits[-1] = '$A_' + tgt_token
            tgt_idx +=1
            continue
        
    for _ in range(src_idx, len(src_tokens)):
        edits.append('$D')
    
    # 如果tgt的下一个是标点，且src处理到的最后一个是KEEP，则改为add
    if tgt_idx < len(tgt_tokens) and tgt_tokens[tgt_idx] in punc_set and edits[-1]=='$K':
        edits[-1] = '$A_' + tgt_tokens[tgt_idx]
        
    return edits


def get_mapping(src_tokens: List[str], tgt_tokens: List[str]) -> List[int]:
    mapping_idx = []
    i, j = 0, 0
    while()

def diff_tokens(src_tokens:List[str], tgt_tokens:List[str]) -> List[str]:
    src_idx, tgt_idx = 0, 0
    s2t_idxs = [] # 源中的token映射到目标token的位置
    t2s_idxs = [] # 目标的token映射到源token的位置
    
    
    
    edits = []
    while src_idx < len(src_tokens) and tgt_idx < len(tgt_tokens):
        src_token = src_tokens[src_idx]
        tgt_token = tgt_tokens[tgt_idx]
        
        # 相等则KEEP
        if src_token == tgt_token:
            edits.append('$K')
            src_idx += 1
            tgt_idx +=1
            continue
        
        # 两个都是标点，则替换
        if src_token in punc_set and tgt_token in punc_set:
            edits.append('$R' + tgt_token)
            src_idx += 1
            tgt_idx +=1
            continue
        
        # src标点，tgt为非标点，则删除，如：我爱北，京 => 我爱北京
        if src_token in punc_set and tgt_token not in punc_set:
            edits.append('$D')
            src_idx += 1
            continue
        
        # src非标点，tgt为标点，则在前一个上标记$A_X，如：我爱北京 => 我爱北，京
        # 则将“北”的标签由“$K”修改为 "$A_,"
        if src_token not in punc_set and tgt_token in punc_set:
            edits[-1] = '$A_' + tgt_token
            tgt_idx +=1
            continue
        
    for _ in range(src_idx, len(src_tokens)):
        edits.append('$D')
    
    # 如果tgt的下一个是标点，且src处理到的最后一个是KEEP，则改为add
    if tgt_idx < len(tgt_tokens) and tgt_tokens[tgt_idx] in punc_set and edits[-1]=='$K':
        edits[-1] = '$A_' + tgt_tokens[tgt_idx]
        
    return edits
        
if __name__ == '__main__':
    src_chars = ['[CLS]'] + list('“我爱北，京天安门天安门上红旗。飞')
    tgt_chars = ['[CLS]'] + list('：“我爱北京天安门，天安门上红：旗飞。')
    edits = diff_token_puncs(src_chars, tgt_chars)
    for a, b in zip(src_chars, edits):
        print(a, ' => ', b)
    print('DONE')