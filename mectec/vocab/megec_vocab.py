import os
import pickle
from typing import List

from logger import logger
from .. import conf
from .. util.lang_util import is_hanzi

class MegecVocab:
    """
    纠错相关的词汇表，包含用于纠正的labels和用于标记是否有错误的少量tags
    """
    def __init__(self) -> None:
        label_file = conf.correct_label_file
        dtag_file = conf.detect_tag_file
        
        if not os.path.exists(label_file):
            raise Exception(f"label vocab file {label_file} does not exist.")
        
        if not os.path.exists(dtag_file):
            raise Exception(f"dtag vocab file {dtag_file} does not exist.")

        #读取数据标签
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = f.readlines()
            labels = [s.strip() for s in labels]
        
        with open(dtag_file, 'r', encoding='utf-8') as f:
            dtags = f.readlines()
            dtags = [s.strip() for s in dtags]
            
        vocab = {}  # 存放标记和检测标签的字典
        vocab["label"] = {w.strip(): i for i, w in enumerate(labels)}        
        vocab["dtag"] = {w.strip(): i for i, w in enumerate(dtags)}
        # 保持不变的纠错标记所对应的id
        self.KEEP_ID =vocab["label"]["$K"]
        self.CORRECT_ID = vocab["dtag"]["CORRECT"]   
        self.vocab = vocab
        self.num_labels = len(labels)
        self.num_dtags = len(dtags)        
        self.labels:List[str] = labels
        self.dtags:List[str] = dtags
                            
    
    def convert_label_to_id(self, label:str, default_label:str="$K") -> int:
        """获取名字为label的纠错标记所对应的id，如果不存在，则返回default_label所对应的
        标记id"""
        id = self.vocab["label"].get(label)
        if id == None:
            id = self.vocab["label"][default_label]
        return id
    
    def convert_id_to_label(self, id:int) -> str:
        return self.labels[id]
    
    def has_label(self, label:str) -> bool:
        """是否存在标记label"""
        return self.vocab["label"].get(label)!=None
                                    
    def convert_ids_to_labels(self, ids:List[int]) -> List[str]:
        """将纠错标签的id转换为名称"""
        return [self.labels[idx] for idx in ids]
                                    
    def convert_dtag_to_id(self, tag:str) -> int:
        """获取名字为tag的检测标记所对应的id"""
        return self.vocab["dtag"].get(tag)
    
    def convert_ids_to_dtags(self, ids:List[int])->List[str]:
        """将检测标签的id，转换为名称，如CORRECT, INCORRECT"""
        return [self.dtags[idx] for idx in ids]
    