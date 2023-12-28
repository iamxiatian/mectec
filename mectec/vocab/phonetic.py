INITIAL_SET = ['b','p', 'm','f','d','t','n','l','g','k',
               'h', 'j','q','x','zh','ch','sh','r','z','c',
               's','y','w']
FINAL_SET = ['a', 'o', 'e', 'i', 'u', 'v', 'ai','ei','ui', 'ao',
             'ou','iu','ie', 've', 'er', 'an', 'en','in','un','vn',
             'ang','eng','ing','ong']




# 易混的声母，由LLM生成的列表
confusing_initials = {  
    'b': ['p'],  
    'p': ['b'],  
    'd': ['t'],  
    't': ['d'],  
    'n': ['l'],  
    'l': ['n'],  
    'g': ['k'],  
    'k': ['g'],  
    'j': ['q'],  
    'q': ['j'],  
    'zh': ['ch'],  
    'ch': ['zh'],  
    'z': ['c'],  
    'c': ['z'],  
    's': ['sh'],  
    'sh': ['s']  
}


# def split_pinyin(pinyin:str)->list[str]:
#     '''将汉字拼音按照声母和韵母分隔开'''
#     shengmu, yunmu = '', pinyin
#     for s in SHENGMU:
#         if pinyin.startswith(s):
#             shengmu = s
#             yunmu = pinyin[len(s):]
#             break
#     return shengmu, yunmu