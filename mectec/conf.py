import logging
import torch
import numpy as np
from transformers import BertTokenizer

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

correct_label_file = './vocabulary/labels.txt'
detect_tag_file = './vocabulary/d_tags.txt'
pinyin_vocab_file = './vocabulary/vocab_pinyin.txt'
pos_vocab_file = './vocabulary/vocab_pos.txt'
glyph_embedding_file = './vocabulary/hanzi_ViT.pkl'

# GEC模型中在句子开始加入的额外Token的名字
GEC_TOKEN_START = '[START]'
GEC_TOKEN_TA = '[TA]'
GEC_TOKEN_DE = '[DE]'
GEC_TOKEN_WHOAMI = '[WHOAMI]'
GEC_TOKEN_DEL = '[DEL]'
BERT_TOKEN_CLS = '[CLS]'
BERT_TOKEN_SEP = '[SEP]'
BERT_TOKEN_PAD = '[PAD]'
BERT_TOKEN_UNK = '[UNK]'
BERT_TOKEN_MASK = '[MASK]'

# 模型可以接受的最大输入长度
MAX_INPUT_LEN = 150

# 模型是否使用拼音信息
#USE_PINYIN = True
# 模型是否使用字形信息
#USE_GLYPH = True

# 是否对特殊字符进行增强
ENHANCE_SPECIAL_CHARS = True 
# 训练用的BATCH_SIZE，训练和推理采用不同数值，预测结果可能会存在差异
BATCH_SIZE = 32

# BERT的路径
#bert_model_path = './pretrain/chinese-roberta-wwm-ext'
#csc_base_path = './pretrain/structbert'
csc_base_path = './pretrain/chinese-roberta-wwm-ext'

SELECT_BASE_MODEL = './pretrain/chinese-roberta-wwm-ext'

# 在计算目标标签时，额外对KEEP标签赋予的权重
keep_confidence:float = 0.0 # 0.3
# 是否需要修改“的地得”时，保留原来大小的可信度
keep_de_confidence:float = 0.0 # 0.9
# 在计算目标标签时，额外对DELETE标签赋予的权重
del_confidence:float = 0.0 # 0.3
# 计算损失时的纠正损失权重
label_loss_weight = 0.2 # 0.25
# 模型预测时，多轮检测的轮次
num_iterations = 1

SKIP_PREDICT_TA = False # 预测时遇到“他她它”的纠错情况，进行忽略。
SKIP_PREDICT_DE = False # 预测时遇到“的地得”的纠错情况，进行忽略。

# 把输入中的的地得更改为[DE]，然后让模型判断此处的具体汉字
CONVERT_INPUT_DE = False
CONVERT_INPUT_TA = False

USE_SELECTOR = False
USE_MLM = False
USE_MLM_TOPN = 10
DTAG_ERROR_P = 0.9 # 检测认为当前位置有错误时，输出概率的最小值限制，小于这个值则认为无错


def normalize_input_text(input:str)->str:
    """对输入进行变换，根据配置，决定是否转换的地得和他她它"""
    s = input
    if CONVERT_INPUT_DE:
        s = s.replace('的', '[DE]').replace('得', '[DE]').replace('地', '[DE]')

    if CONVERT_INPUT_TA:
        s = s.replace('它', '[TA]').replace('他', '[TA]').replace('她', '[TA]')
        
    return s

def fix_seed(seed=17):
    """固定全局种子，以便结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一致


# def _get_tokenizer() -> BertTokenizer:
#     vocab_file = './vocabulary/vocab.txt'
#     # return enhance_tokenzier(BertTokenizer.from_pretrained(vocab_file))
#     return BertTokenizer.from_pretrained(vocab_file)

# BERT的切分器，读取自定义的vocab.txt，里面替换了若干unused字符
bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained('./vocabulary/vocab.txt')