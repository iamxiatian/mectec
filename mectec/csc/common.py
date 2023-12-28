from typing import NamedTuple


# 编辑标签，记录在原始字符串中的起止位置和该位置对应的纠错标签，如:(1,2, $R_中)

class OpEdit(NamedTuple):
    start_pos: int
    end_pos: int
    operation: str

# 描述句子中的一个错误信息
class ErrorItem(NamedTuple):
    pos: int
    type: str
    src: str
    tgt: str
    