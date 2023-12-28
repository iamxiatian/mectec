import copy
import sys

import Levenshtein

from utils.data_helper import is_contain_chinese


def get_spellGCN_labels(src_sentences, tgt_sentences):
    """
    根据原句子和目标句子生成错误信息：
    0 表示没有错误
    标号，错误位置（从1开始计数），推荐词
    C1-1701-2, 19, 持
    C1-1710-1, 2, 题
    C1-1715-1, 52, 地
    C1-1724-1, 11, 疗
    C1-1740-1, 0
    """
    result_list = []
    tgt_list = []
    for sid, src in src_sentences.items():
        item = []
        src_line = src.strip().replace(' ', '').replace("\t", "").replace("	", "")
        tgt_line = tgt_sentences[sid]
        edits = Levenshtein.opcodes(src_line, tgt_line)
        tgt = ""
        for edit in edits:
            if edit[0] == "equal":
                tgt += src_line[edit[1]:edit[2]]
                continue
            wt = src_line[edit[1]:edit[2]]
            et = tgt_line[edit[3]:edit[4]]

            if "。" in et or "U" in et or "U" in wt:
                tgt += wt
                continue
            if "。" in et or "," in et or "," in wt or "“" in wt or "”" in wt:  # rm 。
                tgt += wt
                continue

            if (not is_contain_chinese(wt) and wt != "") or (
                    not is_contain_chinese(et) and et != ""): continue

            tgt += et
            wt_start, wt_end = edit[1], edit[2]
            et_start, et_end = edit[3], edit[4]

            # if len(wt) == len(et) and not self.similar_score(wt, et):
            #     continue

            if edit[0] == "insert":
                err_type = "少字"
                if len(et) >= 2: continue
                # continue
            elif edit[0] == "replace":
                err_type = "替换"
            else:
                err_type = "多字"
                # continue

            result = [wt, et, wt_start, wt_end, et_start, et_end, err_type]
            if item:
                pre_item = item[-1]
                pre_wt, pre_et, pre_wt_start, pre_wt_end, pre_et_start, pre_et_end, pre_err_type = pre_item
                if pre_err_type == "少字" and err_type == "多字" and pre_et == wt \
                        or pre_err_type == "多字" and err_type == "少字" and pre_wt == et:
                    item.pop()
                    wt = src_line[pre_wt_start:wt_end]
                    # 之前为少字错误而当前为多字错误
                    if pre_err_type == "少字" and err_type == "多字":
                        et = pre_et + src_line[pre_wt_end:wt_start]
                    else:
                        et = src_line[pre_wt_end:wt_start] + et
                    err_type = "语序颠倒"
                    result = [wt, et, pre_wt_start, wt_end, pre_et_start, et_end, err_type]
                elif pre_et_end == wt_start:
                    item.pop()
                    wt = src_line[pre_wt_start:wt_end]
                    et = tgt_line[pre_et_start:et_end]
                    result = [wt, et, pre_wt_start, wt_end, pre_et_start, et_end, err_type]
            item.append(result)
        tgt_list.append([sid, tgt])
        item_new = []
        for result in item:
            wt, et, wt_start, wt_end, et_start, et_end, err_type = result
            for i, (wt_, et_) in enumerate(zip(wt, et)):
                result = [str(wt_start + i + 1), et_]
                item_new.append(result)

        out_line = ""
        for res in item_new:
            out_line += ', '.join(res) + ", "
        if out_line:
            # print(sid + ', ' + out_line.strip())
            result_list.append(str(sid) + ', ' + out_line.strip() + "\n")
        else:
            # print(sid + ', -1')
            result_list.append(str(sid) + ', 0' + "\n")

    return result_list


def get_spellGCN_single_labels(sid, src, tgt_line):
    """
    根据原句子和目标句子生成错误信息：
    0 表示没有错误
    标号，错误位置（从1开始计数），推荐词
    C1-1701-2, 19, 持
    C1-1710-1, 2, 题
    C1-1715-1, 52, 地
    C1-1724-1, 11, 疗
    C1-1740-1, 0
    """
    item = []
    src_line = src.strip().replace(' ', '').replace("\t", "").replace("	", "")
    tgt_line = tgt_line.strip().replace(' ', '').replace("\t", "").replace("	", "")
    edits = Levenshtein.opcodes(src_line, tgt_line)
    tgt = ""
    for edit in edits:
        if edit[0] == "equal":
            tgt += src_line[edit[1]:edit[2]]
            continue
        wt = src_line[edit[1]:edit[2]]
        et = tgt_line[edit[3]:edit[4]]

        if "。" in et or "U" in et or "U" in wt:
            tgt += wt
            continue
        if "。" in et or "," in et or "," in wt or "“" in wt or "”" in wt:  # rm 。
            tgt += wt
            continue

        if (not is_contain_chinese(wt) and wt != "") or (
                not is_contain_chinese(et) and et != ""): continue

        tgt += et
        wt_start, wt_end = edit[1], edit[2]
        et_start, et_end = edit[3], edit[4]

        # if len(wt) == len(et) and not self.similar_score(wt, et):
        #     continue

        if edit[0] == "insert":
            err_type = "少字"
            if len(et) >= 2: continue
            # continue
        elif edit[0] == "replace":
            err_type = "替换"
        else:
            err_type = "多字"
            # continue

        result = [wt, et, wt_start, wt_end, et_start, et_end, err_type]
        if item:
            pre_item = item[-1]
            pre_wt, pre_et, pre_wt_start, pre_wt_end, pre_et_start, pre_et_end, pre_err_type = pre_item
            if pre_err_type == "少字" and err_type == "多字" and pre_et == wt \
                    or pre_err_type == "多字" and err_type == "少字" and pre_wt == et:
                item.pop()
                wt = src_line[pre_wt_start:wt_end]
                # 之前为少字错误而当前为多字错误
                if pre_err_type == "少字" and err_type == "多字":
                    et = pre_et + src_line[pre_wt_end:wt_start]
                else:
                    et = src_line[pre_wt_end:wt_start] + et
                err_type = "语序颠倒"
                result = [wt, et, pre_wt_start, wt_end, pre_et_start, et_end, err_type]
            elif pre_et_end == wt_start:
                item.pop()
                wt = src_line[pre_wt_start:wt_end]
                et = tgt_line[pre_et_start:et_end]
                result = [wt, et, pre_wt_start, wt_end, pre_et_start, et_end, err_type]
        item.append(result)
    item_new = []
    for result in item:
        wt, et, wt_start, wt_end, et_start, et_end, err_type = result
        for i, (wt_, et_) in enumerate(zip(wt, et)):
            result = [str(wt_start + i + 1), et_]
            item_new.append(result)

    out_line = ""
    for res in item_new:
        out_line += ', '.join(res) + ", "
    if out_line:
        # print(sid + ', ' + out_line.strip())
        return str(sid) + ', ' + out_line.strip() + "\n"
    else:
        # print(sid + ', -1')
        return str(sid) + ', 0' + "\n"


def get_midu_labels(err_sents, cor_sents):
    """
    :param err_sents dict 原始数据
    :param cor_sents dict 目标数据
    根据原句子和目标句子生成错误信息：
    -1 表示没有错误
    标号，错误位置，错误类型，错别词，推荐词
    0, -1
    1, 9, 别字, 唷, 友,
    """
    result_list = []
    tgt_list = []
    for sid, src in err_sents.items():
        item = []
        src_line_old = copy.deepcopy(src.strip())
        src_line = src.strip().replace(' ', '').replace("\t", "").replace("	", "")
        tgt_line = cor_sents[sid]
        edits = Levenshtein.opcodes(src_line, tgt_line)
        tgt = ""
        for edit in edits:
            if edit[0] == "equal":
                tgt += src_line[edit[1]:edit[2]]
                continue
            wt = src_line[edit[1]:edit[2]]
            et = tgt_line[edit[3]:edit[4]]

            if "。" in et or "U" in et or "U" in wt:
                tgt += wt
                continue
            if "。" in et or "," in et or "," in wt or "“" in wt or "”" in wt:  # rm 。
                tgt += wt
                continue

            if (not is_contain_chinese(wt) and wt != "") or (
                    not is_contain_chinese(et) and et != ""): continue

            tgt += et
            wt_start, wt_end = edit[1], edit[2]
            et_start, et_end = edit[3], edit[4]

            # if len(wt) == len(et) and not self.similar_score(wt, et):
            #     continue

            if edit[0] == "insert":
                err_type = "少字"
                if len(et) >= 2: continue
                # continue
            elif edit[0] == "replace":
                err_type = "替换"
            else:
                err_type = "多字"
                # continue

            result = [wt, et, wt_start, wt_end, et_start, et_end, err_type]
            if item:
                pre_item = item[-1]
                pre_wt, pre_et, pre_wt_start, pre_wt_end, pre_et_start, pre_et_end, pre_err_type = pre_item
                if pre_err_type == "少字" and err_type == "多字" and pre_et == wt \
                        or pre_err_type == "多字" and err_type == "少字" and pre_wt == et:
                    item.pop()
                    wt = src_line[pre_wt_start:wt_end]
                    # 之前为少字错误而当前为多字错误
                    if pre_err_type == "少字" and err_type == "多字":
                        et = pre_et + src_line[pre_wt_end:wt_start]
                    else:
                        et = src_line[pre_wt_end:wt_start] + et
                    err_type = "语序颠倒"
                    result = [wt, et, pre_wt_start, wt_end, pre_et_start, et_end, err_type]
                elif pre_et_end == wt_start:
                    item.pop()
                    wt = src_line[pre_wt_start:wt_end]
                    et = tgt_line[pre_et_start:et_end]
                    result = [wt, et, pre_wt_start, wt_end, pre_et_start, et_end, err_type]
            item.append(result)
        tgt_list.append([sid, tgt])
        item_new = []
        for result in item:
            wt, et, wt_start, wt_end, et_start, et_end, err_type = result
            result = [str(wt_start), err_type, wt, et]
            item_new.append(result)

        out_line = ""
        for res in item_new:
            out_line += ', '.join(res) + ", "
        if out_line:
            # print(sid + ', ' + out_line.strip())
            result_list.append(str(sid) + ', ' + out_line.strip() + "\n")
        else:
            # print(sid + ', -1')
            result_list.append(str(sid) + ', -1' + "\n")

    return result_list


if __name__ == '__main__':
    file = "../data/sighan15.txt"
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    source_list = {}
    target_list = []
    for i, line in enumerate(lines):
        _, src, tgt, _ = line.strip().split("\t")
        source_list[i] = src.strip()
        target_list.append(tgt.strip())

    get_spellGCN_labels(source_list, target_list)
