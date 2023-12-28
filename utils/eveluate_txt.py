import sys

sys.path.append("..")

from utils.get_error_labels import get_midu_labels, get_spellGCN_labels


def read_input_file(input_file):
    """
    获取句子id，原始句子、目标句子、预测句子
    @return 原始句子、目标句子、预测句子对应的字典
    """
    pid_to_src = {}
    pid_to_tgt = {}
    pid_to_pred = {}
    with open(input_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line_list = line.strip().split("\t")
            pid = i
            src = line_list[0]
            tgt = line_list[1]
            pred = line_list[2]
            pid_to_src[pid] = src
            pid_to_tgt[pid] = tgt
            pid_to_pred[pid] = pred
    return pid_to_src, pid_to_tgt, pid_to_pred


def read_label_file(pid_to_text, label_list):
    '''
    获取句子中的错误信息
    :param pid_to_text:字典
    :param label_list:错误标签
    :return:检测错误信息、纠正错误信息
    '''
    error_set, det_set, cor_set = set(), set(), set()
    for line in label_list:
        terms = line.strip().split(',')
        terms = [t.strip() for t in terms]
        pid = int(terms[0])
        if pid not in pid_to_text:
            continue
        if len(terms) == 2 and terms[-1] == '-1':
            continue
        text = pid_to_text[pid].replace(' ', '').replace("\t", "").replace("	", "")
        if (len(terms) - 2) % 4 == 0:
            error_num = int((len(terms) - 2) / 4)
            for i in range(error_num):
                loc, typ, wrong, correct = terms[i * 4 + 1: (i + 1) * 4 + 1]
                loc = int(loc)
                cor_text = text[:loc] + correct + text[loc + len(wrong):]
                error_set.add((pid, loc, wrong, cor_text))
                det_set.add((pid, loc, wrong))
                cor_set.add((pid, cor_text))
        else:
            print('check your data format: {}'.format(line))
            continue
    return error_set, det_set, cor_set


def cal_f1(ref_num, pred_num, right_num):
    try:
        precision = right_num / pred_num if pred_num != 0 else 0
        recall = right_num / ref_num if right_num != 0 else 0
        if precision + recall < 1e-6:
            return 0.0, 0.0, 0.0
        f1 = 2 * precision * recall / (precision + recall)
    except Exception as e:
        return 0, 0, 0
    return recall, precision, f1


def write_labels(pid_to_src, pid_to_tgt, pid_to_pred, ref_labels, pred_labels, save_file):
    update_list = []
    for src, tgt, pred, ref_label, pred_label in zip(pid_to_src.values(), pid_to_tgt.values(), pid_to_pred.values(),
                                                     ref_labels, pred_labels):
        update_list.append([src, tgt, pred, ref_label.strip(), pred_label.strip()])
    update_list = ["\t".join(line) + "\n" for line in update_list]
    with open(save_file, "w", encoding="utf-8") as f:
        f.writelines(update_list)


def evaluate(input_file):
    """
    计算纠错指标
    """
    pid_to_src, pid_to_tgt, pid_to_pred = read_input_file(input_file)
    ref_labels = get_midu_labels(pid_to_src, pid_to_tgt)
    pred_labels = get_midu_labels(pid_to_src, pid_to_pred)

    # write_labels(pid_to_src, pid_to_tgt, pid_to_pred, ref_labels, pred_labels, input_file)

    ref_error_set, ref_det_set, ref_cor_set = read_label_file(pid_to_src, ref_labels)
    pred_error_set, pred_det_set, pred_cor_set = read_label_file(pid_to_src, pred_labels)

    ref_num = len(ref_cor_set)
    pred_num = len(pred_cor_set)

    det_right_num = 0
    for error in ref_error_set:
        pid, loc, wrong, cor_text = error
        if (pid, loc, wrong) in pred_det_set or (pid, cor_text) in pred_cor_set:
            det_right_num += 1
    detect_r, detect_p, detect_f1 = cal_f1(ref_num, pred_num, det_right_num)

    cor_right_num = len(ref_cor_set & pred_cor_set)
    correct_r, correct_p, correct_f1 = cal_f1(ref_num, pred_num, cor_right_num)

    final_score = 0.8 * detect_f1 + 0.2 * correct_f1
    print("detect_p\t {}".format(detect_p))
    print("detect_r\t{}".format(detect_r))
    print("detect_f1\t {}".format(detect_f1))
    print("correct_p\t {}".format(correct_p))
    print("correct_r\t {}".format(correct_r))
    print("correct_f1\t {}".format(correct_f1))
    print("final_score\t {}".format(final_score))
    result_dict = {}
    result_dict["detect_p"] = detect_p
    result_dict["detect_r"] = detect_r
    result_dict["detect_f1"] = detect_f1
    result_dict["correct_p"] = correct_p
    result_dict["correct_r"] = correct_r
    result_dict["correct_f1"] = correct_f1
    result_dict["final_f1"] = final_score
    return result_dict


def evaluate_for_sent(input_file):
    """
    句子级别精准率、召回率、F1值的计算
    """
    inputs_text_list, ref_text_list, pred_text_list = read_input_file(input_file)

    assert len(inputs_text_list) == len(ref_text_list)
    assert len(ref_text_list) == len(pred_text_list)
    total_gold_err, total_pred_err, right_pred_err = len(inputs_text_list), 0, 0

    for sid, ori_tags in inputs_text_list.items():
        ori_tags = ori_tags.lower()
        ref_tags = ref_text_list[sid].lower()
        prd_tags = pred_text_list[sid].lower()

        if ori_tags == ref_tags:
            total_pred_err += 1
        elif ori_tags != prd_tags:
            total_pred_err += 1
        if ref_tags == prd_tags:
            right_pred_err += 1

    print("正报句子:{}".format(right_pred_err))
    print("召回句子：{}".format(total_pred_err))
    # print("误报句子：{}".format())

    p = 1. * right_pred_err / (total_pred_err + 1e-8)
    r = 1. * right_pred_err / (total_gold_err + 1e-8)
    fc = 2 * p * r / (p + r + 1e-8)
    print('sent correction: p=%.3f, r=%.3f, f=%.3f' % (p, r, fc))

    result_dict = {}
    result_dict["cor_sent_num"] = right_pred_err
    result_dict["recall_sent_num"] = total_pred_err
    result_dict["sent_p"] = p
    result_dict["sent_r"] = r
    result_dict["sent_f1"] = fc
    return result_dict


def evaluate_spellGCN(input_file, with_error):
    """
    论文标准测试
    :param input_file: id、错误句子、正确句子、预测句子
    :param with_error: 是否只有错误的答案，true只有错误数据，false错误数据、正确数据都有
    :return:
    """
    detect_TP, detect_FP, detect_FN = 0, 0, 0
    correct_TP, correct_FP, correct_FN = 0, 0, 0
    detect_sent_TP, sent_P, sent_N, correct_sent_TP = 0, 0, 0, 0
    dc_TP, dc_FP, dc_FN = 0, 0, 0
    source_lines, target_lines, pred_lines = read_input_file(input_file)
    pred_labels = get_spellGCN_labels(source_lines, pred_lines)
    actual_labels = get_spellGCN_labels(source_lines, target_lines)
    for idx, (pred, actual) in enumerate(zip(pred_labels, actual_labels)):
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


if __name__ == '__main__':
    # file_name = "sighan15.withError"
    file_name = "dcn.sighan15.txt"
    file = "/data/app/test_data/{}".format(file_name)
    evaluate_spellGCN(file, with_error=False)
    evaluate_for_sent(file)
    evaluate(file)
