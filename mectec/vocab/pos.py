from .vocab import Vocab
from .. import conf
from mectec.util.nlp import tag_text

SPECIAL_POS_TOKENS = ['[PAD]', '[START]', '[MASK]', '[CLS]', '[UNK]', '[SEP]']

class PosVocab:
    """词性标记的词典"""
    def __init__(self) -> None:
        self.vocab:Vocab =  Vocab.load_vocabulary(conf.pos_vocab_file, unk_token='[UNK]', pad_token='[PAD]')
        self.pad_token:str = conf.BERT_TOKEN_PAD
        self.pad_idx:int = self.vocab[conf.BERT_TOKEN_PAD]
        self.cls_idx:int = self.vocab[conf.BERT_TOKEN_CLS]
        self.sep_idx:int = self.vocab[conf.BERT_TOKEN_SEP]
        self.start_idx:int = self.vocab[conf.GEC_TOKEN_START]

    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, tokens):
        return self.vocab[tokens]
    
    def convert_pos_to_ids(self, pos_seq:list[str], max_len=180, padding=False) -> list[str]:
        """把pinyin序列转换为对应的拼音id序列"""
        pos_ids = [self.vocab.token_to_idx[pos] for pos in pos_seq]
        # 用PAD补齐
        if padding and len(pos_seq) < max_len:
            pos_ids.extend([self.vocab.token_to_idx[conf.BERT_TOKEN_PAD]] * (max_len - len(pos_ids)))
        return pos_ids[:max_len]


    def convert_text_to_pos_ids(self, text:str) -> list[int]:
        """根据text，调用分词，获取词性，根据词性得到id，返回每一个字符所对应的词性"""
        tokens = conf.bert_tokenizer.tokenize(text)
        return self.convert_tokens_to_ids(tokens)
    
    
    def convert_tokens_to_ids(self, tokens:list[str]) -> list[int]:
        """把token序列转换为词性id序列"""
        ids = []
        # 注意：不要直接用''.join(tokens)，因为有 sub-word tokenization 的情况
        words, pos_terms = tag_text(conf.bert_tokenizer.convert_tokens_to_string(tokens))
        token_idx = 0
        for word, pos_name in zip(words, pos_terms):
            while token_idx < len(tokens):
                token = tokens[token_idx]
                token = token[2:] if token[:2]=='##' else token
                if token in word:
                    # 每个字符都需要记录对应的词性id，构成词语的每一个字符的词性id相同
                    ids.append(self.vocab.token_to_idx[pos_name])
                    token_idx += 1
                else: break
        assert len(ids) == len(tokens)
        return ids