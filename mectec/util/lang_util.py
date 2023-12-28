from pypinyin import pinyin, Style
import Levenshtein

def hanzi_all(s) -> bool:
    """
    传入的字符串，如果全部都是汉字，返回True，否则返回false。
    注意：标点符号不算汉字
    """
    return all(is_hanzi(ch) for ch in s)

def is_hanzi(ch) -> bool:
    return u'\u4e00' <= ch <= u'\u9fff'

def is_cjk(ch) -> bool:
    if len(ch)!=1: return False
    cp = ord(ch)
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    return ((cp >= 0x4E00 and cp <= 0x9FFF) or 
            (cp >= 0x3400 and cp <= 0x4DBF) or  
            (cp >= 0x20000 and cp <= 0x2A6DF) or  
            (cp >= 0x2A700 and cp <= 0x2B73F) or  
            (cp >= 0x2B740 and cp <= 0x2B81F) or  
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  
            (cp >= 0x2F800 and cp <= 0x2FA1F))

def single_hanzi_to_pinyin(hanzi) -> list[str]:
    """获取单个汉字的拼音，不带音调。如果不是单个汉字，则返回原字符串"""
    if not is_hanzi(hanzi):
        return hanzi
    # 获取拼音
    pinyin_candidates = pinyin(hanzi, style=Style.TONE3, 
                               neutral_tone_with_five=True, 
                               heteronym=True)[0]
    return [candidate[:-1] if candidate[-1].isdigit() else candidate 
                            for candidate in pinyin_candidates]


def get_transform_edits(a:list[str], b:list[str]) -> list:
    """
    把一个字符串转换为另一个字符串操作需要采取的编辑动作，返回类型
    ('delete', 0, 1, 0, 0)
    ('equal', 1, 3, 0, 2)
    ('insert', 3, 3, 2, 3)
    ('replace', 3, 4, 3, 4)
    
    示例：
    >>> get_transform_edits('我爱北京天门安.', '我爱北京天安门.')

    返回：
    [('equal', 0, 5, 0, 5),
    ('replace', 5, 6, 5, 6),
    ('replace', 6, 7, 6, 7),
    ('equal', 7, 8, 7, 8)]
    """
    codes = Levenshtein.opcodes(a, b)

    # 如果遇到两个连续
    final_codes = []
    i = 0
    while i < len(codes):
        # '我爱北京天安门。', '我爱京北天安门。'
        # [('equal', 0, 2, 0, 2),
        # ('insert', 2, 2, 2, 3),
        # ('equal', 2, 3, 3, 4),
        # ('delete', 3, 4, 4, 4),
        # ('equal', 4, 8, 4, 8)]
        # 需要根据上面的情况，将插入、删除修改为替换
        if i<len(codes)-2 and codes[i][0] =='insert' \
            and codes[i+1][0] == 'equal' and codes[i+2][0] == 'delete':

            i11, i12, j11,j12 = codes[i][1], codes[i][2], codes[i][3], codes[i][4]
            i21, i22, j21,j22 = codes[i+1][1], codes[i+1][2], codes[i+1][3], codes[i+1][4]
            i31, i32, j31,j32 = codes[i+2][1], codes[i+2][2], codes[i+2][3], codes[i+2][4]
            if i11==i12 and j11+1==j12 and i21==i11 and i22==i12+1 and j21==j11+1 and j22==j12+1 and i31==i21+1 and i32==i22+1 and j31==j21+1 and j32==j22:
                final_codes.extend(
                    (
                        ('replace', i11, i11 + 1, j11, j11 + 1),
                        ('replace', i31, i32, j21, j22),
                    )
                )
                i += 3
                continue
        final_codes.append(codes[i])
        i += 1
    return final_codes
