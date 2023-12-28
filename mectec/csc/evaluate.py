from typing import NamedTuple, List
import pandas as pd
import json
from .common import ErrorItem
from .data_convert import get_gec_errors, SentenceError
from mectec.util.better import BetterFile
from mectec import conf
from mectec.csc.predict import Predictor as GrammarChecker
from xiatian.selector.predict import Predictor as Selector

__all__ = ['evaluate_file']

class EvaluateSentence(NamedTuple):
    id: str
    src: str
    truth: str
    predicted: str


def calculate_F1(P:float, R:float) -> float:
    return 2 * P * R  / (P+R) if P+R > 0 else 0


def predict_file(checker:GrammarChecker, 
                 selector:Selector, 
                 test_file_name, 
                 out_file:str):
    """测试csv格式保存的文本文件，如果指定了selector，会进一步进行过滤"""
    # model.eval()
    test_data = pd.read_csv(test_file_name, sep='\t', header=None, dtype=str)
    ids = test_data[0].tolist()
    src_stcs =  test_data[1].tolist()
    tgt_stcs = test_data[2].tolist()
    
    # 对预测句子进行后处理，得到最终的预测句子列表，保存在predicted_sentences   
    pred_stcs = checker.predict_sentences(src_stcs)        
    assert len(src_stcs) == len(pred_stcs)

    if selector is not None:
        pred_stcs = selector.select(src_stcs, pred_stcs)

    # 把原始句子的id，原始句子，人工校对过的结果，模型预测结果，预测结果是否完全正确 
    # 逐行保存到out_file中
    with open(out_file, "w", encoding="utf-8") as f:
        f.write('ID\t原句\t正确句\t预测句\t')
        f.write(f'原句与正确句相同\t原句与预测句相同\t预测句与正确句相同\n')
        for id, src, tgt, pred in zip(ids, src_stcs, tgt_stcs, pred_stcs):
            same_src_tgt = 'YES' if src == tgt else 'NO'
            same_src_pred = 'YES' if src == pred else 'NO'
            same_tgt_pred = 'YES' if tgt == pred else 'NO'
            f.write(f'{id}\t{src}\t{tgt}\t{pred}\t{same_src_tgt}')
            f.write(f'\t{same_src_pred}\t{same_tgt_pred}\n')

def read_predicted_file(predicated_file:str, has_header:bool):
    """从预测结果文件中，读取出id，原句，目标句，预测句，并分别放到列表中一起返回"""
    evaluate_sentences:List[EvaluateSentence] = []  
    lines = BetterFile(predicated_file).read_lines()
    start_row = 1 if has_header else 0
    for line in lines[start_row:]:
        items = line.strip().split("\t")
        id, src, truth, predicated = items[:4]
        evaluate_sentences.append(EvaluateSentence(id, src, truth, predicated))
    ids = [e.id for e in evaluate_sentences]
    src_stcs = [e.src for e in evaluate_sentences]
    truth_stcs = [e.truth for e in evaluate_sentences]
    pred_stcs = [e.predicted for e in evaluate_sentences]
    return ids, src_stcs, truth_stcs, pred_stcs

def filter_noises(error_items:List[ErrorItem]) -> List[ErrorItem]:
    """过滤掉SIGHAN中的噪音数据：的地得、她他"""
    filter_items = error_items
    if conf.SKIP_PREDICT_DE:
        filter_items = [
            e for e in filter_items if e.src not in '的地得' or e.tgt not in '的地得'
        ]

    # 代词的处理在预测部分排除 
    #if conf.SKIP_TA:
    #    filter_items = [e for e in filter_items if not (e.src in ('她', '他') and e.tgt in ('她', '他'))]
    return filter_items


def _has_confused_chars(error_items:List[ErrorItem], 
                        confused_group:str) -> bool:
    """是否有的地得等易混错误，方便观察"""
    return any(
        e.src in confused_group and e.tgt in confused_group
        for e in error_items
    )


def evaluate_file(predicated_file:str, 
                  has_header:bool, 
                  save_metrics:bool) -> dict[str, float]:
    """
    读取存放预测结果的文件，计算句子级和词语级的P、R、F指标。
    :param has_header 预测结果文件的第一行是否有字段名称信息
    :param save_metric 是否生成与预测文件同名但多加了.prf后缀的结果文件
    """
    _, src_stcs, truth_stcs, pred_stcs = \
        read_predicted_file(predicated_file, has_header)
    actual_seq:List[SentenceError] = get_gec_errors(src_stcs, truth_stcs)
    pred_seq:List[SentenceError] = get_gec_errors(src_stcs, pred_stcs)

    # 仿照Realise做法，评测时过滤掉噪音数据
    actual_seq = [filter_noises(errors) for errors in actual_seq]
    pred_seq = [filter_noises(errors) for errors in pred_seq]

    # 预测到有错误的句子数量
    pred_error_stc_num = sum(len(s)>0 for s in pred_seq)

    # 实际有错误的句子数量
    true_error_stc_num = sum(len(s)>0 for s in actual_seq)

    # 检测正确的句子数量：必须相同位置，但可以推荐词不同
    def same_location_errors(ts1:List[ErrorItem], ts2:List[ErrorItem]): 
        if len(ts1) != len(ts2) or not ts1: return False
        return all(e1.type == e2.type and e1.pos == e2.pos 
                   for e1, e2 in zip(ts1, ts2))   

    detect_ok_stc_num = sum(same_location_errors(s1, s2) 
                            for s1, s2 in zip(pred_seq, actual_seq)) 

    # 纠正正确的句子数量：位置相同并且推荐词也相同
    correct_ok_stc_num = sum(len(s1)>0 and s1==s2
                             for s1, s2 in zip(pred_seq, actual_seq))

    if pred_error_stc_num==0: pred_error_stc_num=1
    if true_error_stc_num==0: true_error_stc_num=1

    metrics = {}
    metrics['sent_detect_P'] = detect_ok_stc_num / pred_error_stc_num
    metrics['sent_detect_R'] = detect_ok_stc_num / true_error_stc_num
    metrics['sent_detect_F1'] = calculate_F1(metrics['sent_detect_P'], 
                                             metrics['sent_detect_R'])
    
    metrics['sent_correct_P'] = correct_ok_stc_num / pred_error_stc_num
    metrics['sent_correct_R'] = correct_ok_stc_num / true_error_stc_num
    metrics['sent_correct_F1'] = calculate_F1(metrics['sent_correct_P'], 
                                              metrics['sent_correct_R'])

    # 实际标记错误的token数量
    truth_error_token_num = sum(len(ts) for ts in actual_seq)
    pred_error_token_num = sum(len(ts) for ts in pred_seq)
    detect_ok_token_num, correct_ok_token_num = 0, 0
    for ts1, ts2 in zip(pred_seq, actual_seq):
        for e1 in ts1:
            if any(e1.pos==e2.pos and e1.type==e2.type for e2 in ts2):
                detect_ok_token_num += 1
                
            if any(e1.pos==e2.pos and e1.type==e2.type and e1.tgt==e2.tgt 
                   for e2 in ts2):
                correct_ok_token_num += 1

    # 避免除以0错误
    if pred_error_token_num==0: pred_error_token_num = 1
    if truth_error_token_num==0: truth_error_token_num = 1

    metrics['token_detect_P'] = detect_ok_token_num / pred_error_token_num
    metrics['token_detect_R'] = detect_ok_token_num / truth_error_token_num
    metrics['token_detect_F1'] = calculate_F1(metrics['token_detect_P'], 
                                              metrics['token_detect_R'])
    
    metrics['token_correct_P'] = correct_ok_token_num / pred_error_token_num
    metrics['token_correct_R'] = correct_ok_token_num / truth_error_token_num
    metrics['token_correct_F1'] = calculate_F1(metrics['token_correct_P'], 
                                               metrics['token_correct_R'])

    if(save_metrics):
        with open(f"{predicated_file}.prf", "w") as f: 
            json.dump(metrics, f, indent=4)

        # 记录的地得等相关错误
        lines = ['原句\t正确句\t预测句\t预测正常\t他她它\t的地得']
        i = 0
        for actual_errors, pred_errors in zip(actual_seq, pred_seq):
            has_de = _has_confused_chars(actual_errors, '的地得') \
                or _has_confused_chars(pred_errors, '的地得') 
            has_ta = _has_confused_chars(actual_errors, '他她它') \
                or _has_confused_chars(pred_errors, '他她它')
            if has_de or has_ta:
                s = f'{src_stcs[i]}\t{truth_stcs[i]}\t{pred_stcs[i]}'
                s += f'\t{truth_stcs[i]==pred_stcs[i]}\t{has_ta}\t{has_de}'
                lines.append(s)
            i += 1
        BetterFile(f'{predicated_file}.special').save_lines(lines)

    return metrics


def save_human_read_file(predicated_file: str, out_file:str) -> None:
    """将预测结果转换为更适合阅读的格式，保存到out_file文件中"""
    ids, src_stcs, true_stcs, pred_stcs = read_predicted_file(predicated_file,
                                                              True)  
    true_error_items = get_gec_errors(src_stcs, true_stcs)
    pred_error_items = get_gec_errors(src_stcs, pred_stcs)
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for id, src, truth, pred, truth_errors, pred_errors in zip(ids, \
            src_stcs, true_stcs, pred_stcs, true_error_items, \
                pred_error_items):
            truth_tip = '||'.join([
                f'{e.pos},{e.src},{e.tgt}' for e in truth_errors
                ])
            predict_tip = '||'.join([
                f'{e.pos},{e.src},{e.tgt}' for e in pred_errors
                ])
            f.write(f'ID:\t{id}\n')
            f.write(f'Source:\t{src}\n')
            f.write(f'Truth:\t{truth}\t{truth_tip}\n')
            f.write(f'Predict:\t{pred}\t{predict_tip}\n')


def generate_sighan_eval_file(predicated_file:str, 
                              out_file:str, 
                              has_header:bool, 
                              actual_or_predict:str) -> dict[str, float]:
    """
    读取存放预测结果的文件(包含id，原始句子, 正确句子，预测句子），生成sighan的评测文件
    Args:
        predicated_file: 模型预测的结果文件
        out_file: 存放输出结果的文件
        has_header: 预测结果文件的第一行是否有字段名称信息
        actual_or_predict(str): 字符串，如果是actual，则生成实际的纠错结果，
                        如果是predict，则输出预测的处理结果
    Returns:
        返回示例
            句子id,0
            句子id,1,睡,2,觉  
    """
    ids, src_stcs, true_stcs, pred_stcs = read_predicted_file(predicated_file, 
                                                              has_header)
    actual_error_tokens = get_gec_errors(src_stcs, true_stcs)
    pred_error_tokens = get_gec_errors(src_stcs, pred_stcs)
    error_tokens_list = actual_error_tokens if actual_or_predict == 'actual' \
        else pred_error_tokens
        
    lines = []
    for i in range(len(ids)):
        if error_tokens := error_tokens_list[i]:
            items = [f'{e.pos+1},{e.tgt}' for e in error_tokens]
            lines.append(f'{ids[i]},{",".join(items)},')
        else:
            lines.append(f'{ids[i]},0')
    BetterFile(out_file).save_lines(lines)
