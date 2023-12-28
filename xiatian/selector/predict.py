from typing import NamedTuple
from tqdm import tqdm

from mectec import conf
from mectec.util.lang_util import get_transform_edits
from mectec.lm import select_preds

from . import load_model
from .example import generate_examples, make_inputs

class OpEdit(NamedTuple):
    start_pos: int
    end_pos: int
    op: str # operation
    

def _get_edits(src_tokens: list[str], tgt_tokens: list[str]) -> list[OpEdit]:
    """
    根据句子对(src_tokens, tgt_tokens)进行对齐， 生成纠错标签labels和差错标签tags, 原句子会补充上[START]标签
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
    return edits


class Predictor:
    """
    指定模型进行文本纠错的预测处理。初始化时需要指定加载的模型。更简便的方式
    是使用Predictor.load()类方法，如下：
    
    predictor = Predictor.load('./pretrain/chinese-roberta-wwm-ext', 
                                './output/best.bin',
                                device,
                                batch_size=8
                                )
    """

    def __init__(self, device, model, batch_size) -> None:
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.tokenizer = conf.bert_tokenizer

    @classmethod
    def load(cls,
            device,
            ckpt_file,
            batch_size=32):
        model = load_model(ckpt_file)
        model.eval()
        model.to(device)
        return cls(device, model, batch_size)

    def predict_one(self, source: str, target: str):
        """预测错误到纠错结果的句子对，对于识别的纠错结果，通过模型判断是否保留原始结果"""
        
        if source == target: return target
        
        src_tokens = list(source) # conf.bert_tokenizer.tokenize(source)
        tgt_tokens = list(target) # conf.bert_tokenizer.tokenize(target)
        
        edits = _get_edits(src_tokens, tgt_tokens)
        
        examples = generate_examples(src_tokens, tgt_tokens)
        batch = make_inputs(examples)
        token_ids, type_ids, mask = map(
            lambda tensor: tensor.to(self.device), batch)
        pred_labels, scores, _ = self.model(token_ids, type_ids, mask, 
                                            is_test=True)

        # 如果pred_label为1，表示选择就错后的结果，只保留这种情况
        errors:list[OpEdit] = [edit for edit, pred_label in \
                               zip (edits, pred_labels) if pred_label==1]
        
        # 这种情况需要处理：ＤＡＶＩＤ很感谢那位奴生。             
        # ['ｄ', '##ａ', '##ｖ', '##ｉ', '##ｄ', '很', '感', '谢', '那', '位', '奴', '生', '。']
        # 因此，不能用out_tokens = src_tokens， 而是改成：
        # out_tokens = []
        # offset = 0
        # for e in src_tokens:
        #     token_char_num = len(e) 
        #     if e.startswith('##'):
        #         token_char_num = token_char_num - 2
        #     elif e.startswith('[') and e.endswith(']'):
        #         token_char_num = 1
        #     out_tokens.append(source[offset: offset + token_char_num])
        #     offset = offset + token_char_num
            
        outs = src_tokens
        for e in reversed(errors):
            if e.op == '$D':
                outs = outs[:e.start_pos] + outs[e.end_pos:]
            elif e.op.startswith('$A_'):
                outs = outs[:e.start_pos] + [e.op[3:]] + outs[e.end_pos:]
            elif e.op.startswith('$R_'):
                outs = outs[:e.start_pos] + [e.op[3:]] + outs[e.end_pos:]

        return ''.join(outs)
    
    
    def select(self, sources:list[str], targets: list[str]) -> list[str]:
        pairs = zip(sources, targets)
        if len(sources) > 10:
            # 多个句子自动显示进度条
            results = [self.predict_one(s, t) for s, t in \
                    tqdm(pairs, desc="对比句子进行选择...")]
        else:
            results = [self.predict_one(s, t) for s, t in pairs]
            
        if conf.USE_MLM:
            results = select_preds(sources, results)
        return results
