from LAC import LAC
from mectec import conf
# 装载LAC模型
lac = LAC(mode='lac')
lac.add_word('[MASK]/[MASK]')
lac.add_word('[CLS]/[CLS]')
lac.add_word('[UNK]/[UNK]')
lac.add_word('[SEP]/[SEP]')
lac.add_word('[PAD]/[PAD]')
lac.add_word(f'{conf.GEC_TOKEN_START}/{conf.GEC_TOKEN_START}')
lac.add_word(f'{conf.GEC_TOKEN_DE}/{conf.GEC_TOKEN_DE}')
lac.add_word(f'{conf.GEC_TOKEN_TA}/{conf.GEC_TOKEN_TA}')

def tag_text(text:str) -> tuple[list[str], list[str]]:
    lac_result = lac.run(text)
    words, tags = lac_result
    return words, tags
    
def tag_list(texts:list[str]):
    # 批量样本输入, 输入为多个句子组成的list，平均速率更快
    lac_result = lac.run(texts)
    return lac_result[0], lac_result[1]


#print(tag_list('当人生不知该往何处走时，不妨来这岠嵎山，把时间交给自然，让脚步给你答案。'))