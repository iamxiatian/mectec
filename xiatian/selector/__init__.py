import torch as t
from xiatian.selector.modeling import SelectorModel
from xiatian import L

SELECT_FIRST = 0
SELECT_SECOND = 1
SELECT_BOTH = 2

def load_model(last_ckpt_file) -> SelectorModel:
    model = SelectorModel()
    if last_ckpt_file:
        L.info(f'加载已训练模型: {last_ckpt_file}...')
        state_dict = t.load(last_ckpt_file, map_location="cpu")
        # 如果读出来的对象中，存在epoch关键字，则说明训练保存的是某个轮次的中间结果，
        # 对象中的"model"对象的内容是model对象对应的模型参数。
        if "epoch" in state_dict:
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)
    return model
