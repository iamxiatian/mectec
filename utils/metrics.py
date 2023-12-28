from collections import Counter
from typing import List, NamedTuple

def compute_sentence_prf(predict_seq, actual_seq):
    actual_error_sent_num = 0 # 实际的错误句子总数
    predict_error_sent_num = 0 #预测出有错误的句子总数
    detect_TP = 0 # 检测正确的句子总数
    correct_TP = 0 # 纠正正确的句子总数
    
    for predict_labels, actual_labels in zip(predict_seq, actual_seq):
        # 预测句子中是否有错误
        predict_has_error = any(e != "$K" for e in predict_labels) 
        if predict_has_error: predict_error_sent_num += 1

        # 真实句子中是否有错误
        actual_has_error = any(e != "$K" for e in actual_labels)                 
        if actual_has_error: actual_error_sent_num += 1
                
        if predict_has_error and actual_has_error:
            detect_TP += 1
            # 预测结果和实际结果是否完全相同
            if predict_labels == actual_labels:
                correct_TP += 1
            
    metrics = {}
    metrics['detect_P'] = detect_TP / predict_error_sent_num
    metrics['detect_R'] = detect_TP / actual_error_sent_num
    metrics['correct_P'] = correct_TP / predict_error_sent_num
    metrics['correct_R'] = correct_TP / actual_error_sent_num
    return metrics


def compute_p_r_f1(predict_seq, actual_seq):
    """
    计算准确率、召回率、f1
    输入shape: batch_size * seq_len

    注意，序列的每个位置，可能是str，如 '$R_也' 可能是一个list，如 ['$R_就', '$A_是', '$A_于']
    """
    assert len(predict_seq) == len(actual_seq)

    token_detect_TP, token_detect_FP, token_detect_FN = 0, 0, 0
    token_correct_TP, token_correct_FP, token_correct_FN = 0, 0, 0
    sent_detect_TP, sent_detect_FP, sent_detect_FN = 0, 0, 0
    sent_correct_TP, sent_correct_FP, sent_correct_FN = 0, 0, 0

    metrics = {}

    # 统计每个标签的predict_num，actual_num，correct_num
    for predict_labels, actual_labels in zip(predict_seq, actual_seq):
        # predict_labels: 预测的结果列表
        # actual_labels： 实际的结果列表
        assert isinstance(predict_labels, list) and isinstance(actual_labels, list)
        assert len(predict_labels) == len(actual_labels)
        # 处理每一个样本
        for predict_label, actual_label in zip(predict_labels, actual_labels):
            if isinstance(predict_label, str):  # 每个标签都用list表示
                predict_label = [predict_label]
            if isinstance(actual_label, str):
                actual_label = [actual_label]  # 每个标签都用list表示

            predict_types = [] # 预测标签的类型，如$R, $K ...
            for _predict_label in predict_label:
                detect = _predict_label.split('_')[0]
                predict_types.append(detect)
                if detect not in metrics:
                    metrics[detect] = {}
                metrics[detect]['predict_num'] = metrics[detect].get('predict_num', 0) + 1

            actual_types = [] # 真实标签的类型，如$R, $K ...
            for _actual_label in actual_label:
                detect = _actual_label.split('_')[0]
                actual_types.append(detect)
                if detect not in metrics:
                    metrics[detect] = {}
                metrics[detect]['actual_num'] = metrics[detect].get('actual_num', 0) + 1

            # 计算每个标签的检测正确数 detect_correct_num
            # Counter(predict_types) & Counter(actual_types) 表示两个统计结果中重复的情况
            for type_name, cnt in (Counter(predict_types) & Counter(actual_types)).items():
                metrics[type_name]['detect_correct_num'] = metrics[type_name].get('detect_correct_num', 0) + cnt

            # 计算每个标签的 纠正正确数 correct_correct_num
            for detect in [label.split('_')[0] for label in set(predict_label) & set(actual_label)]:
                metrics[detect]['correct_correct_num'] = metrics[detect].get('correct_correct_num', 0) + 1

    # 根据各种num，计算precision、recall、f1
    for label, info in metrics.items():
        # 计算检测p, r, f1
        info['detect_precision'] = info.get('detect_correct_num', 0) / info.get('predict_num', 0) if \
            info.get('predict_num', 0) != 0 else 0
        info['detect_recall'] = info.get('detect_correct_num', 0) / info.get('actual_num', 0) if \
            info.get('actual_num', 0) != 0 else 0
        info['detect_f1'] = 2 * info['detect_precision'] * info['detect_recall'] / (
                info['detect_precision'] + info['detect_recall']) if (
                info['detect_precision'] + info['detect_recall']) != 0 else 0
        # 计算纠错p, r, f1
        info['correct_precision'] = info.get('correct_correct_num', 0) / info.get('predict_num', 0) if \
            info.get('predict_num', 0) != 0 else 0
        info['correct_recall'] = info.get('correct_correct_num', 0) / info.get('actual_num', 0) if \
            info.get('actual_num', 0) != 0 else 0
        info['correct_f1'] = 2 * info['correct_precision'] * info['correct_recall'] / (
                info['correct_precision'] + info['correct_recall']) if (
                info['correct_precision'] + info['correct_recall']) != 0 else 0

    # 计算宏平均
    metrics['mac_avg'] = {}
    label_num = len([label for label in metrics if 'avg' not in label])

    def get_avg_metric(m_name):
        if label_num==0:
            return 0.0
        else:
            return sum([info[m_name] for label, info in metrics.items() if 'avg' not in label]) / label_num

    for metric_name in [
        'detect_precision', 'detect_recall', 'detect_f1', 'correct_precision', 'correct_recall', 'correct_f1'
    ]:
        metrics['mac_avg'][metric_name] = get_avg_metric(metric_name)
    return metrics



PredictRecord = NamedTuple('PredictRecord', [('src_sentence', List[str]), ('tgt_sentence', List[str]), ('pred_sentence', List[str])])


def evaluate_spellGCN(records:List[PredictRecord]):
    """
    论文标准测试
    :param src_sentences: 原句子
    :param with_error: 是否只有错误的答案，true只有错误数据，false错误数据、正确数据都有
    :return:
    """
    detect_TP, detect_FP, detect_FN = 0, 0, 0
    correct_TP, correct_FP, correct_FN = 0, 0, 0
    detect_sent_TP, correct_sent_TP, sent_P, sent_N = 0, 0, 0, 0
    dc_TP, dc_FP, dc_FN = 0, 0, 0

    for src_sentence, tgt_sentence, pred_sentence in records:
        pred_tokens = pred.strip().split(" ")
        actual_tokens = actual.strip().split(" ")
        # assert pred_tokens[0] == actual_tokens[0]
        pred_tokens = pred_tokens[1:]
        actual_tokens = actual_tokens[1:]
        detect_actual_tokens = [int(actual_token.strip(",")) \
                                for i, actual_token in enumerate(actual_tokens) if i % 2 == 0]
        correct_actual_tokens = [actual_token.strip(",") \
                                 for i, actual_token in enumerate(actual_tokens) if i % 2 == 1]
        detect_pred_tokens = [int(pred_token.strip(",")) \
                              for i, pred_token in enumerate(pred_tokens) if i % 2 == 0]
        _correct_pred_tokens = [pred_token.strip(",") \
                                for i, pred_token in enumerate(pred_tokens) if i % 2 == 1]

        # Postpreprocess for ACL2019 csc paper which only deal with last detect positions in test data.
        # If we wanna follow the ACL2019 csc paper, we should take the detect_pred_tokens to:
        max_detect_pred_tokens = detect_pred_tokens

        correct_pred_zip = zip(detect_pred_tokens, _correct_pred_tokens)
        correct_actual_zip = zip(detect_actual_tokens, correct_actual_tokens)

        if detect_pred_tokens[0] != 0:
            sent_P += 1
            if sorted(correct_pred_zip) == sorted(correct_actual_zip):
                correct_sent_TP += 1
        if detect_actual_tokens[0] != 0:
            if sorted(detect_actual_tokens) == sorted(detect_pred_tokens):
                detect_sent_TP += 1
            sent_N += 1

        if detect_actual_tokens[0] != 0:
            detect_TP += len(set(max_detect_pred_tokens) & set(detect_actual_tokens))
            detect_FN += len(set(detect_actual_tokens) - set(max_detect_pred_tokens))
        detect_FP += len(set(max_detect_pred_tokens) - set(detect_actual_tokens))

        correct_pred_tokens = []
        # Only check the correct postion's tokens
        for dpt, cpt in zip(detect_pred_tokens, _correct_pred_tokens):
            if dpt in detect_actual_tokens:
                correct_pred_tokens.append((dpt, cpt))

        correction_list = [actual.split(" ")[0].strip(",")]
        for dat, cpt in correct_pred_tokens:
            correction_list.append(str(dat))
            correction_list.append(cpt)

        correct_TP += len(set(correct_pred_tokens) & set(zip(detect_actual_tokens, correct_actual_tokens)))
        correct_FP += len(set(correct_pred_tokens) - set(zip(detect_actual_tokens, correct_actual_tokens)))
        correct_FN += len(set(zip(detect_actual_tokens, correct_actual_tokens)) - set(correct_pred_tokens))

        # Caluate the correction level which depend on predictive detection of BERT
        dc_pred_tokens = zip(detect_pred_tokens, _correct_pred_tokens)
        dc_actual_tokens = zip(detect_actual_tokens, correct_actual_tokens)
        dc_TP += len(set(dc_pred_tokens) & set(dc_actual_tokens))
        dc_FP += len(set(dc_pred_tokens) - set(dc_actual_tokens))
        dc_FN += len(set(dc_actual_tokens) - set(dc_pred_tokens))

    detect_precision = detect_TP * 1.0 / (detect_TP + detect_FP + 1e-8)
    detect_recall = detect_TP * 1.0 / (detect_TP + detect_FN + 1e-8)
    detect_F1 = 2. * detect_precision * detect_recall / ((detect_precision + detect_recall) + 1e-8)

    correct_precision = correct_TP * 1.0 / (correct_TP + correct_FP + 1e-8)
    correct_recall = correct_TP * 1.0 / (correct_TP + correct_FN + 1e-8)
    correct_F1 = 2. * correct_precision * correct_recall / ((correct_precision + correct_recall) + 1e-8)

    dc_precision = dc_TP * 1.0 / (dc_TP + dc_FP + 1e-8)
    dc_recall = dc_TP * 1.0 / (dc_TP + dc_FN + 1e-8)
    dc_F1 = 2. * dc_precision * dc_recall / (dc_precision + dc_recall + 1e-8)
    result_dict = {}

    if with_error:
        # Token-level metrics
        print("detect_precision=%f, detect_recall=%f, detect_Fscore=%f" % (
            detect_precision, detect_recall, detect_F1))
        print(
            "correct_precision=%f, correct_recall=%f, correct_Fscore=%f" % (
                correct_precision, correct_recall, correct_F1))
        print("dc_joint_precision=%f, dc_joint_recall=%f, dc_joint_Fscore=%f" % (dc_precision, dc_recall, dc_F1))
        result_dict["det_p"] = detect_precision
        result_dict["det_r"] = detect_recall
        result_dict["det_F1"] = detect_F1
        result_dict["cor_p"] = correct_precision
        result_dict["cor_r"] = correct_recall
        result_dict["cor_F1"] = correct_F1
        # result_dict["joint_p"] = dc_precision
        # result_dict["joint_r"] = dc_recall
        # result_dict["joint_F"] = dc_F1

    detect_sent_precision = detect_sent_TP * 1.0 / (sent_P + 1e-8)
    detect_sent_recall = detect_sent_TP * 1.0 / (sent_N + 1e-8)
    detect_sent_F1 = 2. * detect_sent_precision * detect_sent_recall / (
            (detect_sent_precision + detect_sent_recall) + 1e-8)

    correct_sent_precision = correct_sent_TP * 1.0 / (sent_P + 1e-8)
    correct_sent_recall = correct_sent_TP * 1.0 / (sent_N + 1e-8)
    correct_sent_F1 = 2. * correct_sent_precision * correct_sent_recall / (
            (correct_sent_precision + correct_sent_recall) + 1e-8)

    if not with_error:
        # Sentence-level metrics
        print("detect_sent_precision=%f, detect_sent_recall=%f, detect_Fscore=%f" % (
            detect_sent_precision, detect_sent_recall, detect_sent_F1))
        print("correct_sent_precision=%f, correct_sent_recall=%f, correct_Fscore=%f" % (
            correct_sent_precision, correct_sent_recall, correct_sent_F1))

        result_dict["det_sent_p"] = detect_sent_precision
        result_dict["det_sent_r"] = detect_sent_recall
        result_dict["det_sent_F1"] = detect_sent_F1
        result_dict["cor_sent_p"] = correct_sent_precision
        result_dict["cor_sent_r"] = correct_sent_recall
        result_dict["cor_sent_F1"] = correct_sent_F1
    return result_dict