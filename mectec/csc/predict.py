from typing import List, NamedTuple, Union, Tuple
import torch
import pandas as pd
from tqdm import tqdm

from mectec.vocab import megec_vocab
from .data_convert import make_predict_input
from .collate import PadCollate
import Levenshtein
from mectec.util.lang_util import hanzi_all
from mectec.util import BetterString
from mectec import conf
from mectec.csc.loader import load_gec_model

from mectec.lm import mlm_select

TransformAction = NamedTuple('TransformAction', [('start_pos',int), ('end_pos', int),  ('suggest', str), ('prob', float)])

# 将预测的批数据，转换为模型的输入数据
_collate = PadCollate(has_target=False)

def skip(src, tgt):
    return bool(conf.SKIP_PREDICT_TA and src in ('她', '他') and tgt in ('她', '他'))
        
class Predictor:
    """
    指定模型进行文本纠错的预测处理。初始化时需要指定加载的模型。更简便的方式
    是使用Predictor.load()类方法，如下：
    
    predictor = Predictor.load(
        './pretrain/chinese-roberta-wwm-ext', 
        './model/megec/pytorch_model_8',
        'megec', 
        device,
        batch_size=8,
        min_error_probability=min_error_probability,
        min_token_probability=min_token_probability,
        num_iterations = 3
        )
    """
    
    def __init__(self, 
                 model_name, 
                 model, 
                 device, 
                 tokenizer, 
                 batch_size = conf.BATCH_SIZE,
                 min_error_probability=0.0, 
                 min_token_probability=0.0) -> None:
        """
        Args:
            model_name: 模型名称，默认为模型所在的路径
            model: 纠错模型
            device：运行设备(torch.device)
            tokenizer: 切分器，用于输入句子进行分割
            batch_size: 句子数量很多时，会分组进行纠错
            min_stc_prob: 句子错误概率，低于该值则不纠错
            min_label_prob: 纠错词的最低概率，建议词低于该值，则不修改
        """
        self.model_name = model_name
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.min_stc_prob = min_error_probability
        self.min_label_prob = min_token_probability
        
    def predict_sentences(self, sentences:List[str]) -> List[str]:
        """
        对一组未分词的句子进行模型预测，会根据迭代参数conf.num_iterations执行多轮预测，
        返回预测后的句子。 如果句子数量很多，会拆分为batch分批次进行处理
        """
        batch_sentences = []
        predicted_sentences = []
        self.model.eval() # evaluate model:
        with torch.no_grad():
            for src in tqdm(sentences, desc="predict sentences..."):
                batch_sentences.append(src)
                # 形成批次进行测试
                if len(batch_sentences) == self.batch_size:
                    output = self._predict_batch_sentences(batch_sentences)
                    predicted_sentences.extend(output)
                    batch_sentences = []

            if batch_sentences:
                output = self._predict_batch_sentences(batch_sentences)
                predicted_sentences.extend(output)
        return predicted_sentences
    
    
    def _predict_batch_sentences(self,
                                 sentences:List[str], 
                                 iter_idx = 1) -> List[str]:
        """
        对一组未分词的句子进行模型预测，句子数量应在在一个batch_size之内，
        会根据迭代参数num_iterations执行多轮预测，返回预测后的句子。
        具体预测处理通过_predict_tokens完成。
        """
        # 用于记录输出结果
        predicted_tokens = [None] * len(sentences)
        
        # 每一行文本转换为token列表
        input_tokens = [self.tokenizer.tokenize(s) for s in sentences] 
        output_tokens, indicators = self._predict_tokens(input_tokens)   
        pred_stcs = [''.join(tokens) for tokens in output_tokens]
        
        # 后过滤处理，去掉特殊符号处的纠错结果
        pred_stcs = self._postfix(sentences, pred_stcs)
        
        if conf.USE_MLM:
            pred_stcs = mlm_select(sentences, pred_stcs, indicators)
        
        # 如果句子有修改，并且指定了迭代轮次，则重复上述流程
        if iter_idx < conf.num_iterations and pred_stcs != sentences:
            return self._predict_batch_sentences(pred_stcs, iter_idx+1)
        else:
            return pred_stcs
        
    def _to_gpu(self, data):
        if type(data) is torch.Tensor:
            return data.to(self.device) 
        else:
            return None
        
    def _predict_tokens(self, 
                        input_tokens:List[List[str]]
                        ) -> Tuple[List[List[str]], List[List[bool]]]:
        """
        对句子的token序列构成的批次集合，进行预测，输出纠正后的句子的Token序列，
        预测得到的token序列，也没有补充[CLS]和[START]等特殊token
        
        Returns:
            - 纠正结果的token序列
            - 检测结果的0、1序列，0无错误，1有错误（但不一定能纠正出来）
        """
        if not input_tokens: return []
        
        # 对每一个句子的词条序列，转换为输入，tokens中增加了[CLS]、[START]和[SEP]
        batch = [make_predict_input(self.tokenizer, tokens) 
                    for tokens in input_tokens]
        batch_data = _collate(batch)
        batch_data = map(self._to_gpu, batch_data)
        token_ids, pinyin_ids, mask, type_ids, _ = batch_data
                
        output_dict = self.model(token_ids, type_ids, mask,
                                 pinyin_ids = pinyin_ids,
                                 is_test=True)                      
        max_vals = torch.max(output_dict['class_probabilities_labels'], dim=-1)
        label_probs = max_vals[0].tolist()
        label_ids = max_vals[1].tolist()
        max_error_probs = output_dict['max_error_probability']
        
        dtag_probs = output_dict['class_probabilities_d_tags']
        # 注意，batch_tokens中的每一个句子，没有[CLS]和[START]两个标记，
        # 但是其他参数都带有该标记，
        # gen_target_tokens返回时，也不带[CLS]和[START]两个标记
        corrected_tokens = self._to_corrected_tokens(input_tokens, 
                                        label_probs, 
                                        label_ids, 
                                        max_error_probs)
        # 获取每个句子的长度
        n_tokens = [len(a) for a in input_tokens]
        # indicator指示每个位置的dtag是否发现错误
        #indicators = torch.argmax(dtag_probs, dim=-1)
        #indicators = indicators.tolist()
        #indicators = [a[2:2+l] for a, l in zip(indicators, n_tokens)]
        dtag_probs = dtag_probs.tolist()
        dtag_probs = [a[2:2+l] for a, l in zip(dtag_probs, n_tokens)]
        indicators = [[a[0] < conf.DTAG_ERROR_P for a in s ] \
                        for s in dtag_probs]
        return corrected_tokens, indicators
    
    def _to_corrected_tokens(self, 
                             input_tokens, 
                             batch_label_probs, 
                             batch_label_ids, 
                             batch_max_error_probs) -> List[List[str]]:
        """
        根据保存在batch_label_ids中的预测结果，得到预测后的新句子所对应的token序列,
        此处input_tokens中每一个句子的token序列，由原始句子切分形成，不带[CLS]和
        [START]两个特殊token，但batch_label_ids等模型输出结果带有这些参数，需要在
        返回时删除，返回正常纠错后句子的token序列。
        """
        predict_results = []
        for tokens, label_probs, label_ids, max_err_prob in \
            zip(input_tokens, 
                batch_label_probs,
                batch_label_ids,  
                batch_max_error_probs):
            l = min(len(tokens) + 2, conf.MAX_INPUT_LEN)
            
            # 如果句子错误的最大概率小于min_error_probability参数，
            # 或者最大的label标记为0，则不需要纠错，直接记录原始文本的token结果
            if max_err_prob < self.min_stc_prob or max(label_ids[:l]) == 0:
                predict_results.append(tokens)
                continue

            edits = []
            # label_ids的第0个是[CLS]的预测结果，第1个是[START]的预测结果，
            # 需要从[START]位置开始看编辑变换
            for i in range(1, l):
                # 如果没有错误（$K）标签，则忽略
                if label_ids[i] == megec_vocab.KEEP_ID: continue                
                
                # [START]对应的结果不是插入标签$A_XXX，则忽略
                label = megec_vocab.convert_id_to_label(label_ids[i])
                if i == 1 and not label.startswith('$A_'): continue
                
                action = self._get_action(label, index=i, prob=label_probs[i])
                if action is not None: edits.append(action)
                
            # targets用于保存目标预测结果的token
            targets =  [conf.BERT_TOKEN_CLS, conf.GEC_TOKEN_START ] + tokens[:]
            for edit in reversed(edits):
                start, end, suggest, _ = edit
                targets[start: end] = [suggest]    
            # 删除预测结果中的开始的[CLS]和[START]两个标记
            predict_results.append(targets[2:]) 
        return predict_results
    
    def _get_action(self,label, index, prob)-> Union[TransformAction, None]:
        """Get suggested actions for token."""
        # cases when we don't need to do anything
        if prob < self.min_label_prob or label in ['[UNK]', '$K']:
            return None

        # 如果是的地得，则需要大于配置文件规定的最小值
        if (label[:-1] in '的地得' or label[:-1] in '他她它') \
            and prob < conf.keep_de_confidence:
            return None

        # 确定编辑动作所影响的字符串起止位置
        if label.startswith('$R_') or label == '$D' or label == '$T':
            start_pos = index
            end_pos = index + 1
        elif label.startswith("$A_"):
            start_pos = index + 1
            end_pos = index + 1

        # 建议词语
        if label == "$D":
            suggest = ''
        elif label.startswith("$T"):
            suggest = label[:]
        else:
            suggest = label[label.index('_') + 1:]
            
        return TransformAction(start_pos, end_pos, suggest, prob)

    def _postfix(self, src_sentences, corrected_sentences):
        """过滤掉非汉字的纠正结果，对于非汉字情况下的纠正，保留原来句子中的字符"""
        fixed_sentences = [] # 对特殊符号进行过滤修正，得到的最终句子结果
        for src_stc, pred_stc in zip(src_sentences, corrected_sentences):
            fixed_sentence = "" # 保存修正后的句子
            src_pronouns, tgt_pronouns = [], [] # 遇到的代词错误
            src_line = src_stc.strip().replace(' ', '')\
                .replace("\t", "").replace("	", "")
            edits = Levenshtein.opcodes(src_line, pred_stc)
            for edit in edits:
                edit_type, src_start, src_end, tgt_start, tgt_end = edit
                if edit_type == "equal":
                    fixed_sentence += src_line[src_start:src_end]
                    continue
                                
                src_word = src_line[src_start:src_end]
                tgt_suggest = pred_stc[tgt_start:tgt_end]
                # 如果是纠正前或者纠正后的词语，不是汉字，则保留原句子的词语
                if not hanzi_all(src_word) or not hanzi_all(tgt_suggest) \
                    or skip(src_word, tgt_suggest):
                    #if not hanzi_all(src_word) or not hanzi_all(tgt_suggest):
                    fixed_sentence += src_word
                else:            
                    fixed_sentence += tgt_suggest
                    
            #     # 记录他她的位置映射信息，以便纠错后，看句子是否包含了性别信息
            #     if src_word in '他她' and tgt_suggest in '他她':
            #         src_pronouns.append((src_word, src_start))
            #         tgt_pronouns.append((tgt_suggest, len(fixed_sentence)-1))
                    
            # # 上一步修改的“他她”，应该判断目标句子中是否有性别相关的信息
            # if src_pronouns and \
            #     not BetterString(fixed_sentence).has_gender_char():
            #     chars = list(fixed_sentence)
            #     # 没有性别信息，就不要修改，还原原来的人称代词
            #     for (src_char, _), (_, tgt_idx) in zip(src_pronouns, tgt_pronouns):
            #         chars[tgt_idx] = src_char
            #     fixed_sentence = ''.join(chars)
            fixed_sentences.append(fixed_sentence)
            
        return fixed_sentences
    
    
    def predict_file(self, test_file_name, out_file:str):
        """测试csv格式保存的文本文件"""
        # model.eval()
        test_data = pd.read_csv(test_file_name, sep='\t', 
                                header=None, dtype=str)
        ids = test_data[0].tolist()
        src_stcs = test_data[1].tolist()
        tgt_stcs = test_data[2].tolist()
        
        # 对预测句子进行后处理，得到最终的预测句子列表，保存在predicted_sentences   
        pred_stcs = self.predict_sentences(src_stcs)        
        assert len(src_stcs) == len(ids) and len(src_stcs) == len(pred_stcs)

        # 把原始句子的id，原始句子，人工校对过的结果，模型预测结果，预测结果是否完全正确
        # 逐行保存到out_file中
        with open(out_file, "w", encoding="utf-8") as f:
            f.write('ID\t原句\t正确句\t预测句')
            f.write(f'\t原句与正确句相同\t原句与预测句相同\t预测句与正确句相同\n')
            for id, src, tgt, pred in zip(ids, src_stcs, tgt_stcs, pred_stcs):
                same_src_tgt = 'YES' if src == tgt else 'NO'
                same_src_pred = 'YES' if src == pred else 'NO'
                same_tgt_pred = 'YES' if tgt == pred else 'NO'
                f.write(f'{id}\t{src}\t{tgt}\t{pred}\t{same_src_tgt}')
                f.write(f'\t{same_src_pred}\t{same_tgt_pred}\n')
    
    @classmethod    
    def load(cls, 
                base_model_path, 
                model_path:str, 
                model_type:str, 
                device,
                batch_size:int = conf.BATCH_SIZE,
                min_error_probability:float=0.0, 
                min_token_probability:float=0.0):
        model = load_gec_model(base_model_path, model_path, model_type, 0,
                               label_loss_weight=conf.label_loss_weight)
        model.eval()
        model.to(device)
        return cls(model_path, model, 
                   device, 
                   conf.bert_tokenizer, 
                   batch_size, 
                   min_error_probability, 
                   min_token_probability)

    