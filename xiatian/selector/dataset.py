from transformers import BertTokenizer
from torch.utils.data import Dataset
from mectec.data import PairFile

from . example import generate_examples, make_input

TopN = None

class FileSelectorDataset(Dataset):
    """
    存储在文本文件中的数据集
    """
    
    def __init__(self, 
                data_file,
                tokenizer:BertTokenizer, 
                has_line_id:bool = True):
        '''
        对指定的文件进行内容读取，构建数据集
        :param data_file: 训练数据或测试数据
        :param tokenizer_path: 预训练模型路径或vocab.txt路径
        :param enhance_sample: 是否对样本进行增强，将的地得和他她用MASK遮盖掉预测，以便学习上下文信息
        :param vocab_path: 标签路径，包含d_tags.txt和labels.txt
        :param pinyin_path: 拼音路径
        :param has_id: 文件中每一行的开始，是否有句子的id
        :param tag_strategy: keep_one 多个操作型label保留一个
        '''
        super(FileSelectorDataset, self).__init__()
        self.tokenizer = tokenizer
        pair_file = PairFile(data_file, tokenizer, has_line_id)
        tokenized_pairs = pair_file.get_tokenized_pairs(TopN)
        
        self.samples = []
        # 根据句子对，生成训练样本，一个句子可能会有多个错误，则生成过个样本 
        for src, tgt in tokenized_pairs:
            examples = generate_examples(src, tgt)
            for example in examples:
                token_ids, type_ids, mask = make_input(example, False)
                self.samples.append((token_ids, type_ids, mask, 0))
        
                token_ids, type_ids, mask = make_input(example, True)
                self.samples.append((token_ids, type_ids, mask, 1))
                

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回值：token_ids, type_ids, attention_mask, labels
        """
        return self.samples[idx]
