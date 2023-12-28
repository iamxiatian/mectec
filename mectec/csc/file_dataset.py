import random
from tqdm import tqdm
from xiatian import L
from torch.utils.data import Dataset
from .data_convert import align_tokens, make_train_input
from mectec.util import BetterString, BetterFile
from mectec import conf

TopN = None # 获取文件中的前面多少行，如果为None，则取所有内容，测试时，可以设为50

class FileDataset(Dataset):
    '''
    存储在文本文件中的数据集
    '''
    
    def __init__(self, 
                data_file,
                tokenizer, 
                enhance_sample: bool,
                has_id:bool = True):
        '''
        对指定的文件进行内容读取，构建数据集
        Args:
            data_file(str): 训练数据或测试数据
            tokenizer(BertTokenizer): 预训练模型路径或vocab.txt路径
            enhance_sample(bool): 是否对样本进行增强
            has_id(bool): 文件中每一行的开始，是否有句子的id
        '''
        super(FileDataset, self).__init__()
        self.data_file = data_file
        self.tokenizer = tokenizer       
        
        self.enhance_sample = enhance_sample
        
        self.pairs = [] # 切分后的句子对，在load_from_file中完成加载
        self.load_from_file(has_id)


    def enhance_text(self, text:str) -> str:
        """
        对输入进行变换，转换“的地得”和“他她它”：
        - “的地得”和“他她它”有1/4的概率会被替换为其他对应汉字。
        - 第一个“他她它”，只有含有性别相关的词语时，才有50%的概率被替换
        """
        
        chars = list(text)

        string = BetterString(text)

        # 选择“的地得”的位置，按一定概率替换
        char_index_pairs:list[(str, int)] = string.locate_chars('的地得')
        for ch, idx in char_index_pairs:
            # candidates = [c for c in ['的', '地', '得'] if c!=ch]
            # 每个“的地得”有1/4的概率被替换为其他“的地得”
            candidates = [c for c in ['的', '地', '得', ch, ch, ch, ch, ch] ]
            chars[idx] = random.choice(candidates)

        char_index_pairs:list[(str, int)] = string.locate_chars('他她它')
        # 第一个TA，只有可以替换的信息，才会替换，其他情况不替换
        if char_index_pairs and string.has_gender_char():
            ch, idx = char_index_pairs[0]
            # candidates = [c for c in ['她', '他', '它'] if c!=ch]
            # 50%的概率被替换为其他“他她它”
            candidates = [c for c in ['她', '他', '它', ch]]
            chars[idx] = random.choice(candidates)


        # 有两个以上的他她它的时候，后面的TA按概率替换
        for ch, idx in char_index_pairs[1:]:
            # 每个TA有1/4的概率被替换为其他TA
            candidates = [c for c in ['她', '他', '它', ch, ch, ch, ch, ch]]
            chars[idx] = random.choice(candidates)

        return ''.join(chars)
    
    
    def load_from_file(self, has_id:bool):
        """ 从data_file中读取句子对，如果has_id=true, 则需要忽略每一行的第一个字段id """
        random.seed(0)
        self.pairs = []
        L.info(f'load dataset from {self.data_file}')
        for line in tqdm(BetterFile(self.data_file).read_lines(top=TopN)):
            part_num = len(line.strip().split("\t"))
            if part_num < 2 or part_num < 3 and has_id: continue

            src, tgt = line.strip().split("\t")[1:3] \
                            if has_id else line.strip().split("\t")[:2]
            pair = (self.tokenizer.tokenize(src), self.tokenizer.tokenize(tgt))
            self.pairs.append(pair)
            
            if not self.enhance_sample: continue
            
            # 如果两个句子不一样，则保留正确句子到正确句子的样本, 让模型学习到更多的正确信息
            if src != tgt:
                pair = (self.tokenizer.tokenize(tgt), 
                        self.tokenizer.tokenize(tgt))
                self.pairs.append(pair)
            
            if not conf.ENHANCE_SPECIAL_CHARS: continue
            
            # 把的地得和他她它替换掉后，增强数据，强迫模型学习上下文信息
            enhanced_text = self.enhance_text(src)
            if enhanced_text != pair[0]:
                e = (self.tokenizer.tokenize(enhanced_text), pair[1])
                self.pairs.append(e)
                
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tokens, tgt_tokens = self.pairs[idx]
        
        # 注意，此处的src_tokens中，已经在开始位置增加了一个“[START]”词元，
        # 以便能应对少字情况并和纠错标记与检测标记的数量对齐，但没有补上[CLS]
        tokens, _, label_ids, dtag_ids = align_tokens(tokens, tgt_tokens)
        
        return make_train_input(tokenizer=self.tokenizer,
                                tokens=tokens,
                                label_ids=label_ids,
                                dtag_ids=dtag_ids)
