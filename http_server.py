import os
import json
import torch
import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel
from mectec.csc.predict import Predictor
from xiatian.selector.predict import Predictor as Selector
from mectec.csc.evaluate import evaluate_file, predict_file
from mectec import conf
from mectec.lm import mask_choose, select_preds, fill_mask

    
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda:0')


models = {
    "final_gold": {
        "base_model": './pretrain/chinese-roberta-wwm-ext',
        "gec_model": './model/final_gold2/best_stc_12_162000.pt',
        "type": 'gold2'
    },
    "final": {
        "base_model": './pretrain/chinese-roberta-wwm-ext',
        "gec_model": './model/final_best/best_stc_25_169000.pt',
        "type": 'final'
    }
}

current = models["final"]

predictor = Predictor.load(current["base_model"],
                            current["gec_model"],
                            current["type"],
                            device)

#selector =  Selector.load(device, './model/slb1217/select_epoch_10.pt')

#selector = Selector(device, None, 32)
selector = None

app = FastAPI()

cpu_num = 1
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)


@app.get("/")
def hello_world():
    return {"Hello World!"}

class ConfRequest(BaseModel):
    """系统配置信息，方便动态修改"""
    min_error_probability:float = predictor.min_stc_prob
    min_token_probability: float = predictor.min_label_prob
    num_iterations:int = conf.num_iterations 
    keep_confidence:float = conf.keep_confidence
    keep_de_confidence:float = conf.keep_de_confidence
    del_confidence:float = conf.del_confidence
    skip_de:bool = conf.SKIP_PREDICT_DE
    skip_ta:bool = conf.SKIP_PREDICT_TA
    convert_de:bool = conf.CONVERT_INPUT_DE
    convert_ta:bool = conf.CONVERT_INPUT_TA
    use_selector:bool = conf.USE_SELECTOR
    use_mask:bool = conf.USE_MLM
    use_mask_topn:int = conf.USE_MLM_TOPN
    batch_size:int = predictor.batch_size
    dtag_error_p:float = conf.DTAG_ERROR_P

def current_conf():
    return {
        "min_error_probability": predictor.min_stc_prob,
        "min_token_probability": predictor.min_label_prob,
        "num_iterations": conf.num_iterations,
        "keep_confidence": conf.keep_confidence,
        "keep_de_confidence": conf.keep_de_confidence,
        "del_confidence": conf.del_confidence,
        "skip_de": conf.SKIP_PREDICT_DE,
        "skip_ta": conf.SKIP_PREDICT_TA,
        "use_selector": conf.USE_SELECTOR,
        "use_mask": conf.USE_MLM,
        "use_mask_topn": conf.USE_MLM_TOPN,
        "batch_size": predictor.batch_size,
        "dtag_error_p": conf.DTAG_ERROR_P
        }

@app.get("/conf")
def conf_get():
    return Response(json.dumps(current_conf(), ensure_ascii=True), 
                    media_type="application/json")

@app.post("/conf")
def conf_post(req: ConfRequest):
    predictor.min_stc_prob = req.min_error_probability
    predictor.min_label_prob = req.min_token_probability
    predictor.batch_size = req.batch_size
    conf.keep_confidence = req.keep_confidence
    conf.keep_de_confidence = req.keep_de_confidence
    conf.del_confidence = req.del_confidence
    conf.num_iterations = req.num_iterations
    conf.SKIP_PREDICT_DE = req.skip_de
    conf.SKIP_PREDICT_TA = req.skip_ta
    conf.USE_SELECTOR = req.use_selector
    conf.USE_MLM = req.use_mask
    conf.USE_MLM_TOPN = req.use_mask_topn
    conf.DTAG_ERROR_P = req.dtag_error_p
    
    return Response(json.dumps(current_conf(), ensure_ascii=True), 
                    media_type="application/json")


class GecRequest(BaseModel):
    sentences: list[str]

@app.post("/correct")
def correct(req: GecRequest):
    stcs = req.sentences
    pred_stcs = predictor.predict_sentences(stcs)
    if selector:
        pred_stcs = selector.select(stcs, pred_stcs)
    if conf.USE_MLM:
        pred_stcs = select_preds(stcs, pred_stcs)
    response = {
        "result" : pred_stcs,
        "config": current_conf()
    }
    return Response(json.dumps(response, ensure_ascii=False), 
                    media_type="application/json")


class EvaluateRequest(BaseModel):
    eval_file:str

@app.post("/predict_file")
def evaluate(req: EvaluateRequest):
    if conf.USE_SELECTOR:
        predict_file(predictor, selector, req.eval_file, 'predicated.tsv')
    else:
        predict_file(predictor, None, req.eval_file, 'predicated.tsv')
    scores = evaluate_file('predicated.tsv', True,  True)
    response = {
        "model_name": predictor.model_name,
        "eval_file": req.eval_file,
        "scores": scores,
        "config": current_conf(),
    }
    return Response(json.dumps(response, ensure_ascii=True), 
                    media_type="application/json")


class IndicatorRequest(BaseModel):
    predicated_file:str

@app.post("/calculate_prf")
def evaluate(req: IndicatorRequest):
    result = evaluate_file(req.predicated_file, True, True)
    return Response(
        json.dumps(result, ensure_ascii=True), 
        media_type="application/json")


class MaskRequest(BaseModel):
    masked_sentence:str
    candidates:list[str]

@app.post("/mask_choose")
def mask_choose_req(req: MaskRequest):
    chosen_token = mask_choose(req.masked_sentence, req.candidates)
    return Response(chosen_token, media_type="text/plain")

class FillMaskRequest(BaseModel):
    masked_sentences:list[str]
    topn:int
    
@app.post("/fill_mask")
def fill_mask_req(req: FillMaskRequest):
    filled_words = [fill_mask(s, topn=req.topn) for s in req.masked_sentences]
    result = [f'{s}{words}' for s, words in \
        zip(req.masked_sentences, filled_words)]
    return Response('\n\n'.join(result), media_type="text/plain")

def start():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == '__main__':
    conf.fix_seed(100)
    start()

