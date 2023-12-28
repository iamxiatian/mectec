import torch
from logger import logger

from .model_gector import MectecGECToR
from .model_pinyin import MectecPinyin
from .model_glyph import MectecGlyph
from .model_merge import MectecMerge
from .model_final import MectecFinal


def load_gec_model(base_model_path,
                   trained_model_path,
                   model_type,
                   dropout_rate,
                   label_loss_weight
                   ):
    """根据基础模型，纠错模型和模型类型，加载模型"""
    if model_type == 'gector':
        model = MectecGECToR(base_model_path, dropout_rate, label_loss_weight)
    elif model_type == 'final':
        model = MectecFinal(base_model_path, dropout_rate, label_loss_weight)
    elif model_type == 'pinyin':
        model = MectecPinyin(base_model_path, dropout_rate, label_loss_weight)
    elif model_type == 'glyph':
        model = MectecGlyph(base_model_path, dropout_rate, label_loss_weight)
    elif model_type == 'merge':
        model = MectecMerge(base_model_path, dropout_rate, label_loss_weight)

    if trained_model_path:
        logger.info(f'加载已训练模型: {trained_model_path}...')
        state_dict = torch.load(trained_model_path, map_location="cpu")
        # 如果读出来的对象中，存在epoch关键字，则说明训练保存的是某个轮次的中间结果，
        # 对象中的"model"对象的内容是model对象对应的模型参数。
        if "epoch" in state_dict:
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)
    return model
