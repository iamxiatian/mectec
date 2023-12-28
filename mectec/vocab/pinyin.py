from . vocab import Vocab
from .. import conf
from .. util.lang_util import is_hanzi
from pypinyin import lazy_pinyin, Style

class PinyinVocab:
    def __init__(self) -> None:
        pinyin_vocab_file = conf.pinyin_vocab_file
        self.vocab:Vocab =  Vocab.load_vocabulary(pinyin_vocab_file, 
                                                  unk_token='[UNK]', 
                                                  pad_token='[PAD]')
        self.pad_token = '[PAD]'
        self.pad_id = self.vocab['[PAD]']

    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, tokens):
        return self.vocab[tokens]
    
    def convert_tokens_to_pinyins(self, tokens:list[str]) -> list[str]:
        """
        把token序列转换为对应的不包含音调的拼音序列，例如
        tokens: ['hello', 'kitty', ',', '游', '玩', '重', '庆',  '之', 
            '后', '，', '身', '体', '超', '重', '了', '。', 'ok', '?']
        result: ['hello', 'kitty', ',', 'you', 'wan', 'chong', 'qing', 'zhi', 
            'hou', '，', 'shen', 'ti', 'chao', 'zhong', 'le', '。', 'ok', '?']
        """
        s = ''.join(tokens)
        pinyins = lazy_pinyin(s, style=Style.TONE3, 
                                neutral_tone_with_five=True)
        
        token_idx = 0
        result = []
        
        for pinyin in pinyins:
            # 如果当前处理的token是汉字，则记录对应的汉字拼音到result中
            if is_hanzi(tokens[token_idx]) and pinyin != tokens[token_idx]:
                pinyin = pinyin[:-1] if pinyin[-1].isdigit() else pinyin
                result.append(pinyin)
                token_idx += 1
                continue
            
            # 跳过组成pinyin变量的所有token
            start_idx = token_idx
            
            result.append(tokens[token_idx])
            token_idx += 1
            while ''.join(tokens[start_idx:token_idx]) != pinyin:
                result.append(tokens[token_idx])
                token_idx += 1
                if token_idx == len(tokens): break
            if token_idx == len(tokens): break
        return result

    def convert_pinyin_to_ids(self, pinyin_seq:list[str]) -> list[str]:
        """把pinyin序列转换为对应的拼音id序列"""
        return [self.vocab.token_to_idx[pinyin] for pinyin in pinyin_seq]

    def convert_tokens_to_ids(self, 
                              tokens:list[str], 
                              max_len=conf.MAX_INPUT_LEN, 
                              padding=False) -> list[int]:
        """把token序列转换为对应的拼音id的序列"""
        pinyin_seq = self.convert_tokens_to_pinyins(tokens)
        pinyin_ids = self.convert_pinyin_to_ids(pinyin_seq)                 
        # 用PAD补齐
        if padding and len(pinyin_ids) < max_len:
            pinyin_ids.extend([self.pad_id] * (max_len - len(pinyin_ids)))
        return pinyin_ids[:max_len]
