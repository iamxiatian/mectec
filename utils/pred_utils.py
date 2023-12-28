import copy

import Levenshtein
import torch

from mectec.vocab import glyph_vocab
from utils.data_helper import remove_space, check, is_contain_chinese
import torch.nn as nn
import torch.nn.functional as F
from pypinyin import lazy_pinyin, Style
import numpy as np


# def segment_data(tokenizer, test_data):
#     """
#     对数据进行分词
#     """
#     update_list = []
#     for line in test_data:
#         tokens = tokenizer.tokenize(line)
#         update_list.append(tokens)
#     return update_list


def processor_data_to_bert(tokenizer, tokens_list, max_len, device):
    """
    将数据进行预处理，处理成bert可预测格式
    @param tokenizer bert分词器
    @param tokens_list 分词后的token列表
    @param max_len 最大长度
    @param device 指定GPU
    return  token_ids_list, attention_mask_list, segment_ids_list
    """
    token_ids_list, attention_mask_list, segment_ids_list = [], [], []
    for token_word in tokens_list:
        token_word = ["[CLS]"] + ['$START'] + token_word + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(token_word)
        token_len = len(token_ids)
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            attention_mask = [1] * max_len
        else:
            token_ids += [0] * (max_len - token_len)
            attention_mask = [1] * token_len + [0] * (max_len - token_len)
        segment_ids = [0] * max_len
        token_ids_list.append(token_ids)
        attention_mask_list.append(attention_mask)
        segment_ids_list.append(segment_ids)

    token_ids_list = torch.tensor(token_ids_list, dtype=torch.long).to(device)
    attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.long).to(device)
    segment_ids_list = torch.tensor(segment_ids_list, dtype=torch.long).to(device)

    return token_ids_list, attention_mask_list, segment_ids_list


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def processor_data_to_bert_pinyin(tokenizer, tokens_list, max_len, pinyin_vocab, device, glyph_vocab=None):
    """
    将数据进行预处理，处理成bert可预测格式
    @param tokenizer bert分词器
    @param tokens_list 分词后的token列表
    @param max_len 最大长度
    @param device 指定GPU
    return  token_ids_list, attention_mask_list, segment_ids_list
    """
    token_ids_list, attention_mask_list, segment_ids_list, pinyin_ids_list, char_emb_list = [], [], [], [], []
    for token_word in tokens_list:
        token_word = ["[CLS]"] + ['$START'] + token_word + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(token_word)
        token_len = len(token_ids)
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
            attention_mask = [1] * max_len
        else:
            token_ids += [0] * (max_len - token_len)
            attention_mask = [1] * token_len + [0] * (max_len - token_len)
        segment_ids = [0] * max_len

        # Use pad token in pinyin emb to map word emb [CLS], [SEP]
        if pinyin_vocab:
            pinyins = lazy_pinyin(
                token_word[2:-1], style=Style.TONE3, neutral_tone_with_five=True)
            pinyin_ids = [0, 0]
            # Align pinyin and chinese char
            # 对于长度不为1的字符(不太理解这种情况)或不为中文的字符 将pinyin_vocab['UNK']或pinyin['PAD']添加至pinyin_ids
            pinyin_offset = 0
            for i, word in enumerate(token_word[2:-1]):
                pinyin = '[UNK]' if word != '[PAD]' else '[PAD]'
                if len(word) == 1 and is_chinese_char(ord(word)):
                    while pinyin_offset < len(pinyins):
                        current_pinyin = pinyins[pinyin_offset][:-1]
                        pinyin_offset += 1
                        if current_pinyin in pinyin_vocab:
                            pinyin = current_pinyin
                            break
                pinyin_ids.append(pinyin_vocab[pinyin])

            pinyin_ids.append(0)
            if len(pinyin_ids) < max_len:
                pinyin_ids += ([0] * (max_len - len(pinyin_ids)))
            pinyin_ids_list.append(pinyin_ids[:max_len])

        if glyph_vocab:
            char_embeddings = glyph_vocab.get_glyph_embeddings(token_ids, glyph_vocab, max_len)
            char_emb_list.append(char_embeddings)

        token_ids_list.append(token_ids)
        attention_mask_list.append(attention_mask)
        segment_ids_list.append(segment_ids)

    token_ids_list = torch.tensor(token_ids_list, dtype=torch.long).to(device)
    attention_mask_list = torch.tensor(attention_mask_list, dtype=torch.long).to(device)
    segment_ids_list = torch.tensor(segment_ids_list, dtype=torch.long).to(device)

    if glyph_vocab:
        pinyin_ids_list = torch.tensor(pinyin_ids_list, dtype=torch.long).to(device)
        char_emb_list = torch.tensor(char_emb_list, dtype=torch.float).to(device)
        return token_ids_list, attention_mask_list, segment_ids_list, pinyin_ids_list, char_emb_list
    elif pinyin_vocab:
        pinyin_ids_list = torch.tensor(pinyin_ids_list, dtype=torch.long).to(device)

        return token_ids_list, attention_mask_list, segment_ids_list, pinyin_ids_list
    else:
        return token_ids_list, attention_mask_list, segment_ids_list, None


# def _convert(data, model_weights, is_tensor=True):
#     """
#     分析模型预测结果
#     @param data 模型预测结果
#     @param model_weights 模型权重 [1]
#     return  probs:token 级别的预测分数
#             idxs：预测标签
#             error_probs: 句子级别的预测分数
#     """
#     if is_tensor:
#         all_class_probs = torch.zeros_like(data[0]['class_probabilities_labels'])
#         error_probs = torch.zeros_like(data[0]['max_error_probability'])
#         # max_error_index = int(data[0]['max_error_index'])
#         for output, weight in zip(data, model_weights):
#             all_class_probs += weight * output['class_probabilities_labels'] / sum(model_weights)
#             error_probs += weight * output['max_error_probability'] / sum(model_weights)

#         max_vals = torch.max(all_class_probs, dim=-1)
#         probs = max_vals[0].tolist()
#         idxs = max_vals[1].tolist()
#     else:
#         all_class_probs = np.zeros_like(data[0]['class_probabilities_labels'])
#         error_probs = np.zeros_like(data[0]['max_error_probability'])
#         for output, weight in zip(data, model_weights):
#             all_class_probs += weight * output['class_probabilities_labels'] / sum(model_weights)
#             error_probs += weight * output['max_error_probability'] / sum(model_weights)

#         probs = np.max(all_class_probs, axis=-1).tolist()
#         idxs = np.argmax(all_class_probs, axis=-1).tolist()
#     return probs, idxs, error_probs.tolist()


def apply_edits(source_tokens, edits):
    """
    根据原始句子和标签，得到预测句子
    @param source_tokens 分词后的token
    @param edits 标签
    """
    target_tokens = source_tokens[:]
    shift_idx = 0
    for edit in edits:
        start, end, label, _ = edit
        target_pos = start + shift_idx
        source_token = target_tokens[target_pos] \
            if len(target_tokens) > target_pos >= 0 else ''
        if label == "":
            del target_tokens[target_pos]
            shift_idx -= 1
        elif start == end:
            word = label.replace("$A_", "")
            target_tokens[target_pos: target_pos] = [word]
            shift_idx += 1
        elif start == end - 1:
            # if label.startswith("$T"):
            #     target_tokens[target_pos] = "T"
            # else:
            word = label.replace("$R_", "")
            target_tokens[target_pos] = word

    return [target_tokens, edits]


def get_next_predict_batch(last_batch, pred_ids, pred_batch, prev_preds_dict, result_list):
    """
    判断新一轮预测结果仍然能检测出错误的句子
    @param final_batch  上轮需要预测的句子
    @param pred_ids  上轮需要预测的index
    @param pred_batch   上轮预测结果
    @param prev_preds_dict 字典：id：预测结果

    return 返回新一轮更新后的预测结果和下一轮仍需预测的id list
    """
    new_pred_ids = []
    total_updated = 0
    for i, orig_id in enumerate(pred_ids):
        orig = last_batch[orig_id]
        pred = pred_batch[i][0]
        prev_preds = prev_preds_dict[orig_id]
        if orig != pred and pred not in prev_preds:
            edits = pred_batch[i][1]
            last_batch[orig_id] = pred
            new_pred_ids.append(orig_id)
            prev_preds_dict[orig_id].append(pred)
            result_list[orig_id].extend(edits)
            total_updated += 1
        elif orig != pred:
            # update final batch, but stop iterations
            edits = pred_batch[i][1]
            last_batch[orig_id] = pred
            total_updated += 1
            result_list[orig_id].extend(edits)
        else:
            continue
    return last_batch, new_pred_ids, total_updated, result_list


def get_tgt_src(tgt_list):
    tgt_list_new = []
    for wt in tgt_list:
        if str(wt).startswith("##"):
            wt = wt.replace("#", "")
        if wt.__eq__("[UNK]"):
            wt = "U"
        tgt_list_new.append(wt)
    return "".join(tgt_list_new)


# PAD = "@@PADDING@@"
# UNK = "@@UNKNOWN@@"
# START_TOKEN = "$START"
# UNK1 = "[UNK]"


# def get_token_action(min_probability, index, prob, correct_label):
#     """Get lost of suggested actions for token."""
#     # cases when we don't need to do anything
#     if prob < min_probability or correct_label in [UNK, PAD, UNK1, '$K']:
#         return None

#     if correct_label.startswith('$R_') or correct_label == '$D' or correct_label == '$T':
#         start_pos = index
#         end_pos = index + 1
#     elif correct_label.startswith("$A_"):
#         start_pos = index + 1
#         end_pos = index + 1

#     if correct_label == "$D":
#         sugg_token_clear = ""
#     elif correct_label.startswith("$T"):
#         sugg_token_clear = correct_label[:]
#     else:
#         try:
#             sugg_token_clear = correct_label[correct_label.index('_') + 1:]
#         except Exception as e:
#             print("出错数据:{}".format(correct_label))

#     return start_pos - 2, end_pos - 2, sugg_token_clear, prob


def convert_from_sentpair_to_edits_new(src_sentences, corrected_sentences):
    fixed_sentences = [] # 对特殊符号进行过滤修正，得到的最终句子结果
    transformed_tokens = [] # 变换过的tokens，方便输出token级别的纠错结果
    for src_sentence, corrected_sentence in zip(src_sentences, corrected_sentences):
        item = [ ]
        src_line = src_sentence.strip().replace(' ', '').replace("\t", "").replace("	", "")
        edits = Levenshtein.opcodes(src_line, corrected_sentence)
        fixed_sentence = "" # 保存修正后的句子
        for edit in edits:
            if edit[0] == "equal":
                fixed_sentence += src_line[edit[1]:edit[2]]
                continue
            src_word = src_line[edit[1]:edit[2]]
            tgt_suggest = corrected_sentence[edit[3]:edit[4]]

            # 如果是纠正前或者纠正后的词语，是标点符号，或者不是汉字，则保留原句子的词语
            
            
            if "。" in tgt_suggest or "U" in tgt_suggest or "U" in src_word:
                fixed_sentence += src_word
                continue
            
            if  "," in tgt_suggest or "," in src_word or "“" in src_word or "”" in src_word: 
                fixed_sentence += src_word
                continue

            if (not is_contain_chinese(src_word) and src_word != "") or (
                    not is_contain_chinese(tgt_suggest) and tgt_suggest != ""):
                fixed_sentence += src_word
                continue

            fixed_sentence += tgt_suggest
            wt_start, wt_end = edit[1], edit[2]
            et_start, et_end = edit[3], edit[4]

            # if len(wt) == len(et) and not self.similar_score(wt, et):
            #     continue

            if edit[0] == "insert":
                err_type = "少字"
                if len(tgt_suggest) >= 2: continue
                # continue
            elif edit[0] == "replace":
                err_type = "替换"
            else:
                err_type = "多字"
                # continue

            result = [src_word, tgt_suggest, wt_start, wt_end, et_start, et_end, err_type]
            if item:
                pre_item = item[-1]
                pre_wt, pre_et, pre_wt_start, pre_wt_end, pre_et_start, pre_et_end, pre_err_type = pre_item
                if pre_err_type == "少字" and err_type == "多字" and pre_et == src_word \
                        or pre_err_type == "多字" and err_type == "少字" and pre_wt == tgt_suggest:
                    item.pop()
                    src_word = src_line[pre_wt_start:wt_end]
                    # 之前为少字错误而当前为多字错误
                    if pre_err_type == "少字" and err_type == "多字":
                        tgt_suggest = pre_et + src_line[pre_wt_end:wt_start]
                    else:
                        tgt_suggest = src_line[pre_wt_end:wt_start] + tgt_suggest
                    err_type = "语序颠倒"
                    result = [src_word, tgt_suggest, pre_wt_start, wt_end, pre_et_start, et_end, err_type]
                elif pre_et_end == wt_start:
                    item.pop()
                    src_word = src_line[pre_wt_start:wt_end]
                    tgt_suggest = corrected_sentence[pre_et_start:et_end]
                    result = [src_word, tgt_suggest, pre_wt_start, wt_end, pre_et_start, et_end, err_type]
            item.append(result)
        fixed_sentences.append(fixed_sentence)
        item_new = []
        for result in item:
            src_word, tgt_suggest, wt_start, wt_end, et_start, et_end, err_type = result
            result = [str(wt_start), err_type, src_word, tgt_suggest]
            item_new.append(result)

        out_line = ""
        for res in item_new:
            out_line += ', '.join(res) + ', '
        if out_line:
            # print(sid + ', ' + out_line.strip())
            transformed_tokens.append(out_line.strip() + "\n")
        else:
            # print(sid + ', -1')
            transformed_tokens.append('-1' + "\n")

    return fixed_sentences, transformed_tokens


def convert_from_sentpair_to_edits(err_sents, cor_sents, char_score_list, type_list):
    """
    type:1-错别字，2-多字，3-少字，4-语序颠倒
    """

    # assert len(cor_sents) == len(err_sents)
    result_list = []
    for i in range(len(cor_sents)):
        item = []
        if len(char_score_list[i]) == 0:
            result_list.append(item)
            continue
        src_line_old = copy.deepcopy(err_sents[i])
        space_num = src_line_old.count(" ") + src_line_old.count("\t") + src_line_old.count("	")
        src_line = remove_space(err_sents[i])
        tgt_line = cor_sents[i]
        edits = Levenshtein.opcodes(src_line, tgt_line)
        err_num = 0
        score_list = char_score_list[i]
        for edit in edits:
            if edit[0] == "equal": continue
            wt = src_line[edit[1]:edit[2]]
            et = tgt_line[edit[3]:edit[4]]

            if "。" in et or "U" in et or "U" in wt: continue
            if "。" in et or "," in et or "," in wt or "“" in wt or "”" in wt:  # rm 。
                continue
            # if check(wt) or check(et): continue
            # if (not is_contain_chinese(wt) and wt != "") or (
            #         not is_contain_chinese(et) and et != ""): continue

            wt_start, wt_end = edit[1], edit[2]
            et_start, et_end = edit[3], edit[4]

            # if len(wt) == len(et) and not self.is_similar_score(wt, et):
            #     continue

            if edit[0] == "insert":
                err_type = "少字"
            elif edit[0] == "replace":
                err_type = "替换"
            else:
                err_type = "多字"

            score_result = score_list[err_num]
            start, _, suggest_word, score = score_result
            if suggest_word == et and start == edit[1]:
                score = round(score, 2)
            else:
                score = 0.5
            err_num += 1
            if score <= 0.2:
                level = -1
            elif score > 0.2 and score <= 0.5:
                level = 0
            else:
                level = 1

            result = [wt, et, wt_start, wt_end, et_start, et_end, err_type, score, level]
            if item:
                pre_item = item[-1]
                pre_wt, pre_et, pre_wt_start, pre_wt_end, pre_et_start, pre_et_end, pre_err_type, pre_score, pre_level = pre_item
                score = (score + pre_score) / 2
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
                    result = [wt, et, pre_wt_start, wt_end, pre_et_start, et_end, err_type, score, level]
                elif pre_et_end == wt_start:
                    item.pop()
                    wt = src_line[pre_wt_start:wt_end]
                    et = tgt_line[pre_et_start:et_end]
                    result = [wt, et, pre_wt_start, wt_end, pre_et_start, et_end, err_type, score, level]
            item.append(result)

        item_new = []
        for result in item:
            wt, et, wt_start, wt_end, et_start, et_end, err_type, score, level = result
            if err_type == "少字" and wt_end < len(src_line) - 1:
                wt = src_line[wt_start: wt_end + 1]
                et = tgt_line[et_start: et_end + 1]
            elif err_type == "少字":
                wt = src_line[wt_start - 1: wt_end]
                et = tgt_line[et_start - 1: et_end]
            try:
                wt_start_ = wt_start - 1 if (wt_start - 1) > 0 else 0
                position = src_line_old.index(wt, wt_start_, wt_start + len(wt) + space_num)
                # if err_type=="少字": continue
                result = {"position": str(position), "type": err_type, "wt": wt, "et": et,
                          "model_type": type_list[i],
                          "det_score": score, "cor_score": score, "level": level}
            except Exception as e:
                print("{}获取位置时出错：{}".format(src_line_old, str(e)))
                continue
            item_new.append(result)

        # tgt_list.append(tgt_line)
        result_list.append(item_new)
    return result_list


def convert_sent_pairs_to_labels(err_sents, cor_sents):
    # assert len(cor_sents) == len(err_sents)
    label_list = []
    for i in range(len(cor_sents)):
        item = []
        src_line = err_sents[i].strip().replace(' ', '').replace("\t", "").replace("	", "")
        tgt_line = cor_sents[i]
        edits = Levenshtein.opcodes(src_line, tgt_line)
        tgt = ""
        for edit in edits:
            if edit[0] == "equal":
                tgt += src_line[edit[1]:edit[2]]
                continue
            wt = src_line[edit[1]:edit[2]]
            et = tgt_line[edit[3]:edit[4]]

            if et == 'U':
                continue
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
            result = [str(wt_start), err_type, wt, et]
            item_new.append(result)

        out_line = ""
        for res in item_new:
            out_line += '， '.join(res) + '， '
        if out_line:
            print(str(i) + '， ' + out_line.strip())
            label_list.append(str(i) + '， ' + out_line.strip() + "\n")
        else:
            print(str(i) + '， -1')
            label_list.append(str(i) + '， -1' + "\n")

    return label_list
