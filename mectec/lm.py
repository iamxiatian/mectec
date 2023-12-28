import torch
from transformers import BertForMaskedLM, BertTokenizer
import pypinyin

from mectec import conf


def get_pinyin(c:str)->list[str]:
    '''返回单个字符的拼音列表（多音字有多个字符构成）'''
    pinyins = pypinyin.pinyin(c, heteronym=True, style=pypinyin.NORMAL)[0]
    return pinyins

def same_pinyin(c1, c2) -> bool:
    '''两个字符是否存在相同的拼音'''
    pinyin1 = get_pinyin(c1)
    pinyin2 = get_pinyin(c2)
    return not set(pinyin1).isdisjoint(pinyin2)

def similar_pinyin(c1, c2) -> bool:
    '''两个字符是否存在相同的拼音'''
    pinyin1 = get_pinyin(c1)
    pinyin2 = get_pinyin(c2)
    # same = False
    # for p1 in pinyin1:
    #     for p2 in pinyin2:
    #         if p1 == p2 or p1.startswith(p2) or p2.startswith(p1):
    #             same = True
    #             break
    same = not set(pinyin1).isdisjoint(pinyin2)
    return same

def find_similar(hanzi:str, candidates:list[str]):
    '''从候选列表汉字中挑选一个和hanzi拼音最相近的结果，找不到返回None'''
    same = False
    
    return True

device = torch.device('cuda')

# 加载模型
mlm = BertForMaskedLM.from_pretrained(conf.csc_base_path).to(device)  
tokenizer = BertTokenizer.from_pretrained(conf.csc_base_path)

def fill_mask(s:str,topn=5) -> list[str]:
    """对句子中的[mask]进行预测，输出TopN个字"""
    tokens = tokenizer.tokenize(s)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.as_tensor([token_ids], dtype=torch.long).to(device)
    output = mlm(token_ids)    
    
    logits = output[0].squeeze()
    mask_token_idx = tokens.index('[MASK]')
    pred_ids = torch.argsort(logits[mask_token_idx], descending=True)[:topn]
    return tokenizer.convert_ids_to_tokens(pred_ids)

    
def mask_choose(masked_string:str, candidate_tokens:list[str]) -> str:
    """对句子中的[mask]进行预测，输出TopN个字"""
    tokens = tokenizer.tokenize(masked_string)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.as_tensor([token_ids], dtype=torch.long).to(device)
    output = mlm(token_ids)    
    logits = output[0].squeeze()

    candidates_idx = tokenizer.convert_tokens_to_ids(candidate_tokens)
    candidates_idx = torch.tensor(candidates_idx).to(device)

    mask_token_idx = tokens.index('[MASK]')
    scores = logits[mask_token_idx].index_select(-1, candidates_idx)
    return candidate_tokens[torch.argmax(scores).item()]  

def select_pred(src_stc, pred_stc):
    '''根据句子差异，把差异出MASK掉，计算MASK距离哪一个更接近，保留接近的结果'''
    src_tokens = list(src_stc) # conf.bert_tokenizer.tokenize(source)
    pred_tokens = list(pred_stc)
    
    best = src_tokens
    for i, (src_token, pred_token) in enumerate(zip(src_tokens, pred_tokens)):
        if src_token != pred_token:
            # 替换为MASK，并进行选择
            best[i] = '[MASK]'
            c = mask_choose(''.join(best), [src_token, pred_token])
            best[i] = c
    return ''.join(best)

def select_preds(src_stcs:list[str], pred_stcs: list[str]) -> list[str]:
    return [select_pred(s, t) for s, t in zip(src_stcs, pred_stcs)]
    
    
def pick_by_mlm(stc:str, pred_stc, indicator:list[bool]):
    '''
    针对原始句子stc，利用MECTEC检测到错误，但没有纠正错误时，继续利用MLM纠错。
    因为只对检测有问题的位置，且纠正没有纠正出来的情况，进行判断，因此，可以提升
    检测效果
    '''
    targets = list(pred_stc)
    for i, (a, b, has_error) in enumerate(zip(stc, pred_stc, indicator)):
        if a==b and has_error:
            # 没有改动，但是检测认为有错误，尝试利用MLM纠错
            targets[i] = '[MASK]'
            words = fill_mask(''.join(targets), conf.USE_MLM_TOPN)
            # print(stc)
            # print(pred_stc)
            # print(''.join(targets), ':', words)
            # print('-----------')
            picked = next((w for w in words if same_pinyin(w, a)), b)
            targets[i] = picked
        elif a!=b and has_error:
            #print(stc)
            #print(pred_stc)
            pass
        elif a !=b and not has_error:
            #print(f'{stc} => {pred_stc}: {a} => {b}')
            pass
            
    return ''.join(targets)

def mlm_select(src_stcs:list[str], 
               pred_stcs:list[str], 
               indicators:list[list[bool]]) -> list[str]:
    '''
    对一组原始句子和预测结果，以及记录在indicators中的检测结果，进行MLM修正
    '''
    result = []
    for src, pred, indicator in zip(src_stcs, pred_stcs, indicators):
        result.append(pick_by_mlm(src, pred, indicator))
        
    return result
    