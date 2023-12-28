class BetterFile:
    def __init__(self, filename:str, encoding:str='utf-8') -> None:
        self.filename = filename
        self.encoding = encoding
        
    def save_lines(self, lines:list):
        with open(self.filename, 'w', encoding=self.encoding) as f:
            f.writelines([f'{line.strip()}\n' for line in lines])
            
    def read_lines(self, top:int=None) -> list[str]:
        with open(self.filename, 'r', encoding=self.encoding) as f:
            lines = f.readlines() if top is None else f.readlines()[:top]
            return [line.strip() for line in lines]


class BetterString:
    def __init__(self, text:str) -> None:
        self.text = text
        
    def has_gender_char(self) -> bool:
        """句子中是否包含性别相关的汉字"""
        return any(ch in self.text 
                   for ch in '男女哥兄姐弟妹爸妈父母公婆爷奶叔伯舅侄甥夫妻姑娘')

    def locate_chars(self, chars:str) -> list[(str, int)]:
        """定位chars中出现的每一个字符，在text中的位置，返回字符及位置构成的对的列表"""
        return [(ch, i) for i, ch in enumerate(self.text) if ch in chars]