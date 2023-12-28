from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
import pandas as pd
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch


processor = DetrImageProcessor.from_pretrained("/data/app/base_model/facebook/detr-resnet-101")
model = DetrForObjectDetection.from_pretrained("/data/app/base_model/facebook/detr-resnet-101")

# 读取vocab
vocab_path = "/data/app/yangyahe/bcm-gector/vocabulary/vocab.txt"
df = pd.read_csv(vocab_path, sep=',,,', encoding='utf_8_sig', header=None,index_col=None, dtype=str)
print(df.shape)
df = df.head()
# print(df.head())
# print(type(df))


# 开始文字转图片
font_path = "/data/app/yangyahe/font/SourceHanSansSC-Regular-2.otf"

def c2p(text):
    im = Image.new("RGB", (500, 500), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    font_size = 300 if isinstance(text, str) and len(text)==1 else 100
    font = ImageFont.truetype(font_path, font_size)
    dr.text((0, 0), str(text), fill="#000000", font=font)
    # im.show()
    inputs = processor(images=im, return_tensors="pt")
    outputs = model(**inputs)
    outputs = torch.sum(outputs["logits"], dim=1).detach().numpy().tolist()
    # file_path = f'/data/app/yangyahe/bcm-gector/data/char_pic/{text}_.png'
    # im.save(file_path)
    return outputs


tqdm.pandas()
df[1] = df[0].progress_apply(c2p)
print(df.head())

char_emb_path = "/data/app/yangyahe/bcm-gector/vocabulary/char_emb.csv"
df.to_csv(char_emb_path, sep='\t', encoding='utf_8_sig', header=False, index=False)
