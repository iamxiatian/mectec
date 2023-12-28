import random
import logging
from tqdm import tqdm

from mectec.conf import bert_tokenizer
from mectec.util import BetterString, BetterFile

class PairFile:
    def __init__(self, data_file:str, tokenizer = None, 
                    has_line_id:bool = True,
                    enhance_sample:bool = False) -> None:
        """封装一个句子对构成的文本文件，文本文件中的每一行是一个样本，由两部分或者三部分构成

        Args:
            data_file (str): 文件名称
            tokenizer (_type_, optional): 句子切分器，没有指定则使用conf.bert_tokenizer
            has_line_id (bool, optional): 每一行的第一部分是否有一个id标记 Defaults to True.
            enhance_sample (bool, optional):是否对句子进行增强变化，目前是处理“的地得”和“他她它”. Defaults to False.
        """
        self.data_file = data_file
        self.has_line_id = has_line_id
        self.enhance_sample = enhance_sample
        self.tokenizer = tokenizer if tokenizer is not None else bert_tokenizer
    
    def _enhance_text(self, text:str) -> str:
        """对输入进行变换，根据配置，决定是否转换的地得和他她它"""
        
        chars = list(text)
        string = BetterString(text)

        # 将的地得随机替换
        char_index_pairs:list[(str, int)] = string.locate_chars('的地得')
        for ch, idx in char_index_pairs:
            candidates = [c for c in ['的', '地', '得'] if c!=ch]
            chars[idx] = random.choice(candidates)
            # chars[idx] = '[MASK]'

        char_index_pairs:list[(str, int)] = string.locate_chars('他她它')
        # 第一个TA，只有可以替换的信息，才会替换，其他情况不替换
        if char_index_pairs and string.has_gender_char():
            ch, idx = char_index_pairs[0]
            candidates = [c for c in ['她', '他', '它'] if c!=ch]
            chars[idx] = random.choice(candidates)


        # 有两个以上的他她它的时候，后面的TA需要替换
        for ch, idx in char_index_pairs[1:]:
            candidates = [c for c in ['她', '他', '它'] if c!=ch]
            chars[idx] = random.choice(candidates)

        #s = s.replace('它', '[MASK]').replace('他', '[MASK]').replace('她', '[MASK]')
        return ''.join(chars)
    
    def get_pair_columns(self, top:int=None) -> tuple[list[str], list[str]]:
        """
        读取文件中的句子对，如果has_line_id=True，则说明每行第一个字段是句子id。
        返回两个数组，第一个数组存放了第一个元素的列表，第二个数组存放了第2个元素的列表。
        """
        random.seed(0)
        column1, column2 = [], []
        
        """ 从data_file中读取句子对，如果has_id=true, 则需要忽略每一行的第一个字段id """
        logging.info(f'Load pair file from {self.data_file}')
        for line in tqdm(BetterFile(self.data_file).read_lines(top)):
            part_num = len(line.strip().split("\t"))
            if part_num < 2 or part_num < 3 and self.has_line_id: continue

            src, tgt = line.strip().split("\t")[1:3] if self.has_line_id else line.strip().split("\t")[:2]
            column1.append(src)
            column2.append(tgt)
        return column1, column2

    def get_tokenized_pairs(self, top:int=None) -> list[(list[str], list[str])]:
        """
        读取文件中的句子对，如果has_line_id=True，则说明每行第一个字段是句子id。
        返回切分后的句子对。
        """
        random.seed(0)
        tokenized_pairs = []
        
        """ 从data_file中读取句子对，如果has_id=true, 则需要忽略每一行的第一个字段id """
        logging.info(f'Load selector train data from {self.data_file}')
        for line in tqdm(BetterFile(self.data_file).read_lines(top)):
            part_num = len(line.strip().split("\t"))
            if part_num < 2 or part_num < 3 and self.has_line_id: continue

            src, tgt = line.strip().split("\t")[1:3] if self.has_line_id else line.strip().split("\t")[:2]
            pair = (self.tokenizer.tokenize(src), self.tokenizer.tokenize(tgt))
            tokenized_pairs.append(pair)
            
            if not self.enhance_sample: continue
            
            # 把的地得和他她它替换掉后，增强数据，强迫模型学习上下文信心
            enhanced_text = self._enhance_text(src)
            if enhanced_text != pair[0]:
                tokenized_pairs.append((self.tokenizer.tokenize(enhanced_text), pair[1]))
        return tokenized_pairs