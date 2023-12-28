import os
import re
from pathlib import Path
import time
from datetime import datetime
from typing import List

import json
from tqdm import tqdm
import pandas as pd
import torch as t
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import (
    RandomSampler,
    DataLoader,
    SequentialSampler,
    DistributedSampler,
)
from transformers import AdamW
from tensorboardX import SummaryWriter

from utils.metrics import compute_p_r_f1

from xiatian import L
from mectec.vocab import megec_vocab
from mectec.csc import FileDataset
from mectec import conf
from mectec.csc import Predictor, PadCollate
from mectec.csc.loader import load_gec_model
from mectec.csc.evaluate import evaluate_file


class Trainer(object):
    def __init__(
        self,
        task_name,
        model_name,
        device,
        max_len=128,
        start_epoch=0,
        epoch_num=30,
        batch_size=32,
        learning_rate=1e-5,
        dropout_rate=0.2,
        base_model="chinese-roberta-wwm-ext",
        correct_f1_weight=0.5,
        label_loss_weight=0.2,
        last_ckpt_file=None,
        parallel=False,
        additional_del_confidence=0,
        additional_confidence=0,
        amp=False,
        local_rank=-1,
        num_workers=4,
        cuda_num=1,
        resume=False,
        seed=0,
        model_type="megec",
        train_path=None,
        val_path=None,
        test_path=None,
    ):
        """
        Args:
            task_name: 训练任务的名称，根据该名字生成模型保存路径和TensorboardX日志
            model_name: 存储模型的名称
            device: cuda
            vocab_path: 词标签路径
            max_len: 句子最大长度
            epoch_num: 训练轮数
            batch_size: 批大小，决定一次训练的样本数目
            learning_rate:学习率
            dropout_rate: 丢弃率
            base_model: 基础模型（bert、robert）
            correct_f1_weight:纠正层所占权重
            last_ckpt_file:上次训练的模型文件
            parallel:是否并行训练
            additional_del_confidence: $D的置信度
            additional_confidence: $K的置信度
            amp: 是否使用混合精度
            local_rank:
            cuda_num:DP训练时，cuda的个数
            resume:是否断点续训
            seed:随机种子数
        """
        super(Trainer, self).__init__()
        self.task_name = task_name
        self.save_dir = f"./model/{task_name}"
        self.model_name = model_name
        self.max_len = max_len
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.base_model = base_model
        self.correct_f1_weight = correct_f1_weight
        self.label_loss_weight = label_loss_weight
        self.last_ckpt_file = last_ckpt_file
        self.parallel = parallel
        self.device = device
        self.local_rank = local_rank
        self.additional_confidence = additional_confidence
        self.additional_del_confidence = additional_del_confidence
        self.amp = amp
        self.cuda_num = cuda_num
        self.tokenizer = conf.bert_tokenizer
        self.model = None
        self.resume = resume
        self.optimizer = None
        self.start_epoch = start_epoch
        self.seed = seed
        self.model_type = model_type
        self.num_workers = num_workers
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path

        self.train_loader, self.val_loader = self._load_data()

        # 创建模型保存的目录
        os.makedirs(self.save_dir, exist_ok=True)

        if self.amp:
            # 在训练最开始之前实例化一个GradScaler对象
            self.scaler = GradScaler()
        self.writer = SummaryWriter(f"./logs/{task_name}")

    def _load_data(self):
        start_time = time.time()
        train_dataset = FileDataset(self.train_path, self.tokenizer, 
                                    enhance_sample=True, 
                                    has_id=True)
        val_dataset = FileDataset(self.val_path, self.tokenizer, 
                                  enhance_sample=False, 
                                  has_id=True)
        L.info(f"CUDA({self.local_rank})加载数据：{time.time() - start_time}s")

        collate_fn = PadCollate(has_target=True)
        if self.local_rank != -1:
            train_sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=train_sampler,
                num_workers=self.num_workers,
                pin_memory=False,
                prefetch_factor=2,
                persistent_workers=True,
                collate_fn=collate_fn,
            )
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=False,
                prefetch_factor=2,
                persistent_workers=True,
                collate_fn=collate_fn,
            )
        else:
            train_sampler = RandomSampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                batch_size=self.batch_size,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            val_sampler = SequentialSampler(val_dataset)
            val_loader = DataLoader(
                val_dataset,
                sampler=val_sampler,
                batch_size=self.batch_size,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        return train_loader, val_loader

    def _load_model(self):
        self.model = load_gec_model(
            self.base_model,
            self.last_ckpt_file,
            self.model_type,
            self.dropout_rate,
            self.label_loss_weight
        )
        self.model.to(self.device)
        if self.local_rank != -1:
            L.info(f"DDP加载模型:{self.local_rank}")
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[int(self.local_rank)],
                output_device=int(self.local_rank),
                find_unused_parameters=True,
            )
        elif self.parallel:
            L.info(f"DP加载模型:{self.local_rank}")
            device_ids = list(range(self.cuda_num))
            self.model = nn.DataParallel(self.model, device_ids=device_ids)

    def _to_gpu(self, data):
        return data.to(self.device) if type(data) is t.Tensor else data

    def train(self, val_step=10000):
        conf.fix_seed(self.seed)

        token_best_models, stc_best_models = [], []

        start_time = time.time()
        self._load_model()
        self._save_config()
        L.info(f"CUDA {self.local_rank} 加载模型耗时{time.time()-start_time}s")

        # 参数组，每组参数可以指定自己的优化器参数，即每组参数可使用不同的优化策略
        param_optimizer = list(self.model.named_parameters())
        # 实现L2正则化接口，对模型中的所有参数进行L2正则处理防止过拟合，包括权重w和偏置b
        # no_decay中存放不进行权重衰减的参数
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        # 判断optimizer_param中所有的参数。如果不在no_decay中，则进行权重衰减;
        # 如果在no_decay中，则不进行权重衰减
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer \
                        if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer \
                        if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        # 使用带有权重衰减功能的Adam优化器Adamw
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        # 学习率调整，如果在验证机上结果变化不大时，应该减小学习率
        scheduler = ReduceLROnPlateau(optimizer, factor=0.35, patience=500)

        global_step, global_loss_total = 0, 0
        best_token_f1 = -1
        best_stc_f1 = -1
        for epoch in range(self.start_epoch, self.epoch_num):
            if self.local_rank != -1:
                self.train_loader.sampler.set_epoch(epoch)

            L.info("开始训练...")
            self.model.train()

            epoch_step, epoch_loss_total = 0, 0

            loop = tqdm(
                self.train_loader, 
                desc=f"Epoch:{epoch},local_rank:{self.local_rank}")
            for batch_data in loop:
                optimizer.zero_grad()  # 梯度置零

                # 把数据移动到指定的device上
                batch_data = map(self._to_gpu, batch_data)
                (
                    input_ids,
                    pinyin_ids,
                    label_ids,
                    dtag_ids,
                    mask,
                    type_ids,
                    _,
                ) = batch_data
                out = self.model(input_ids, type_ids, mask, 
                                 label_ids, dtag_ids, pinyin_ids)

                epoch_step += 1
                global_step += 1

                loss = out["loss_total"]

                if self.amp:
                    # Scales loss. 为了梯度放大.
                    self.scaler.scale(loss).backward()

                    # scaler.step() 首先把梯度的值unscale回来.
                    # 如果梯度值不是infs或NaNs, 那么调用optimizer.step()来更新权重,
                    # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                    self.scaler.step(optimizer)

                    # 准备着，查看是否要增大scaler
                    self.scaler.update()
                else:
                    if self.parallel and self.local_rank == -1:
                        loss = loss.mean()
                    # 反向传播，参数更新
                    loss.backward()
                    optimizer.step()

                if self.local_rank in [-1, 0]:
                    self.writer.add_scalar(
                        "lr",
                        optimizer.state_dict()["param_groups"][0]["lr"],
                        global_step=global_step,
                    )

                    global_loss_total += loss.item()
                    epoch_loss_total += loss.item()

                    epoch_ave_loss = epoch_loss_total / epoch_step
                    loop.set_postfix(loss=epoch_ave_loss)
                    self.writer.add_scalar(
                        "loss", epoch_ave_loss, global_step=global_step
                    )
                    lr = optimizer.state_dict()["param_groups"][0]["lr"]
                    L.info(f"学习率：{lr}, loss:{epoch_ave_loss}")

                    if global_step % val_step == 0 and global_step != 0:
                        self.model.eval()  # 进入测试模式，测试完后要打开训练模式
                        best_token_f1 = self._save_best_token_model(
                            epoch, global_step, best_token_f1,
                            token_best_models,  scheduler)
                        best_stc_f1 = self._save_best_stc_model(
                            epoch, global_step, best_stc_f1, stc_best_models)
                        self.model.train()

            if self.local_rank not in [-1, 0]:
                continue

            L.info(f"Epoch {epoch}已完成，验证...")
            self.model.eval()  # 进入测试模式，测试完后要打开训练模式
            best_token_f1 = self._save_best_token_model(
                epoch, global_step, best_token_f1, token_best_models,
                scheduler)
            best_stc_f1 = self._save_best_stc_model(
                epoch, global_step, best_stc_f1, stc_best_models)
            self._rm_tmp_files() # 删除不需要的预测结果文件
            self.model.train()

    def _save_best_token_model(self, epoch, global_step, 
                               best_token_f1, token_best_models,
                               scheduler) -> float:
        """评价token级别指标，如果是最好结果，则保存，同时返回目前最好的f1指标"""
        new_f1 = best_token_f1
        L.info(f"Token验证中(step: {global_step})...")
        val_label_metric, _, ave_val_loss = self._token_evaluate()
        replace_f1 = (
            0.5 * val_label_metric["$R"]["detect_f1"]
            + 0.5 * val_label_metric["$R"]["correct_f1"]
        )
        self.writer.add_scalar("token_detect_F1",
                               val_label_metric["$R"]["detect_f1"],
                               global_step=global_step)
        self.writer.add_scalar("token_correct_F1", 
                               val_label_metric["$R"]["correct_f1"], 
                               global_step=global_step)
        self.writer.add_scalar("token_F1", replace_f1, global_step=global_step)
        
        if replace_f1 > best_token_f1:
            new_f1 = replace_f1
            model = self.model.module \
                if hasattr(self.model, "module") else self.model
            if self.local_rank in [-1, 0]:
                best_model_name = os.path.join(
                    self.save_dir, 
                    f"best_token_{epoch}_{global_step}.pt"
                )
                t.save(
                    model.state_dict(),
                    best_model_name,
                    _use_new_zipfile_serialization=False,
                )
                token_best_models.append(best_model_name)
                if len(token_best_models) > 3:
                    rm_model = token_best_models.pop(0)
                    os.remove(rm_model)
                log_info = {
                    "step": global_step,
                    "weighted F1": replace_f1,
                    "pred_result": val_label_metric,
                    "saved model": self.model_name,
                }
                fn = os.path.join(self.save_dir, 
                                  "validation_token_record.txt")
                with open(fn, "a+", encoding="utf-8") as f:
                    f.write("\n")
                    f.write(json.dumps(log_info, ensure_ascii=False, indent=4))

        # 根据验证结果更新学习率，scheduler会自动收集历次的ave_val_loss
        scheduler.step(ave_val_loss)
        return new_f1

    def _save_best_stc_model(self, 
            epoch, global_step, best_stc_f1, stc_best_models) -> float:
        """评价句子级别指标，如果是最好结果，则保存，同时返回目前最好的f1指标"""
        new_f1 = best_stc_f1
        metric = self._test_file(epoch, global_step)
        stc_f1 = 0.5 * metric["sent_detect_F1"] \
            + 0.5 * metric["sent_correct_F1"]
        self.writer.add_scalar("stc_detect_P", 
                               metric["sent_detect_P"], 
                               global_step=global_step)
        self.writer.add_scalar("stc_detect_R", 
                               metric["sent_detect_R"], 
                               global_step=global_step)
        self.writer.add_scalar("stc_detect_F1", 
                               metric["sent_detect_F1"], 
                               global_step=global_step)
        self.writer.add_scalar("stc_correct_P", 
                               metric["sent_correct_P"], 
                               global_step=global_step)
        self.writer.add_scalar("stc_correct_R", 
                               metric["sent_correct_R"], 
                               global_step=global_step)
        self.writer.add_scalar("stc_correct_F1", 
                               metric["sent_correct_F1"], 
                               global_step=global_step)
        self.writer.add_scalar("stc_F1", stc_f1, global_step=global_step)
        
        log_info = {"epoch": epoch, 
                    "step": global_step,
                    "test_result": stc_f1, 
                    "test_result_sent": metric}
        fn = os.path.join(self.save_dir, "val_epoch_result.txt")
        with open(fn, "a+", encoding="utf-8") as fp:
            fp.write("\n")
            fp.write(json.dumps(log_info, ensure_ascii=False, indent=4))

        if stc_f1 > best_stc_f1:
            new_f1 = stc_f1
            model_name = os.path.join(
                self.save_dir, f"best_stc_{epoch}_{global_step}.pt"
            )
            stc_best_models.append(model_name)
            if self.local_rank in [-1, 0]:
                L.info("保存DDP模型 CUDA:{}".format(self.local_rank))
                model = self.model.module \
                    if hasattr(self.model, "module") else self.model
                t.save(
                    model.state_dict(), 
                    model_name, 
                    _use_new_zipfile_serialization=False
                )
            if len(stc_best_models) > 3:
                rm_model = stc_best_models.pop(0)
                os.remove(rm_model)
        return new_f1

    def _tensor_to_labels(self, 
                          batch_ids, 
                          seq_len: List[int]) -> List[List[str]]:
        """
        把batch中的id序列，转换为标记名称序列，同时删除第一个[CLS]位置的元素和多余的pad元素
        """
        batch_ids = batch_ids.detach().tolist()
        batch_ids = [ids[1:length] for ids, length in zip(batch_ids, seq_len)]
        return [[megec_vocab.labels[id] for id in ids] for ids in batch_ids]

    def _token_evaluate(self):
        """token级别的验证"""
        val_loader = self.val_loader
        val_loss_total = 0
        all_pred_labels = []  # 所有预测纠错结果对应的标记，如$K,$R_中
        all_pred_dtags = []  # 所有检测结果对应的标记，如CORRECT，INCORRECT
        all_gt_labels = []  # 所有正确纠错结果对应的标记，如$K,$R_中
        all_gt_dtags = []  # 所有正确检测结果对应的标记，如CORRECT，INCORRECT
        with t.no_grad():
            for batch_data in tqdm(val_loader, desc="Evaluating..."):
                batch_data = map(self._to_gpu, batch_data)
                (
                    token_ids,
                    pinyin_ids,
                    label_ids,
                    dtag_ids,
                    mask,
                    type_ids,
                    seq_len,
                ) = batch_data

                out = self.model(token_ids,
                                type_ids,
                                mask,
                                label_ids,
                                dtag_ids,
                                pinyin_ids,
                                is_test=True)
                # 当前批次内句子长度构成的列表，由tensor转为list
                seq_len = seq_len.detach().tolist()

                # 把当前批次中所有的真实纠正标记labels和检测标签tags，
                # 都追加到对应的列表中。
                gt_labels = self._tensor_to_labels(label_ids, seq_len)
                all_gt_labels += gt_labels

                logits_labels = out["logits_labels"]  # 概率分布
                logits_labels = t.argmax(logits_labels, dim=-1)  # 转ID
                predict_labels = self._tensor_to_labels(logits_labels, seq_len)
                all_pred_labels += predict_labels

                loss_total = out["loss_total"].item()
                val_loss_total += loss_total

            # 采用纠正后的结果作为计算检测值和纠正值的依据，比如纠正位置正确，
            # 但纠正结果错误，则表示检测正确，纠正错误
            val_label_metric = compute_p_r_f1(all_pred_labels, all_gt_labels)
            val_dtag_metric = compute_p_r_f1(all_pred_dtags, all_gt_dtags)
            ave_val_loss = val_loss_total / len(val_loader)

        correct_f1 = val_label_metric["mac_avg"]["correct_f1"]
        detect_f1 = val_label_metric["mac_avg"]["detect_f1"]
        final_metric = (
            self.correct_f1_weight * correct_f1
            + (1 - self.correct_f1_weight) * detect_f1
        )
        out_logging = {
            "detailed_label_metric": val_label_metric,
            "val loss": float(ave_val_loss),
            "weighted F1": final_metric,
        }
        if self.local_rank in [0, -1]:
            print(json.dumps(out_logging, ensure_ascii=False, indent=4))
        return val_label_metric, val_dtag_metric, ave_val_loss

    def _test_file(self, epoch, step) -> dict[str, float]:
        """测试csv格式保存的文本文件"""
        # model.eval()
        test_data = pd.read_csv(self.test_path, sep="\t", 
                                header=None, dtype=str)
        ids = test_data[0].tolist()
        src_stcs = test_data[1].tolist()
        tgt_stcs = test_data[2].tolist()

        # 对预测句子进行后处理，得到最终的预测句子列表，保存在predicted_sentences
        predictor = Predictor(
            self.save_dir,
            self.model,
            self.device,
            self.tokenizer,
            batch_size=self.batch_size,
        )
        pred_stcs = predictor.predict_sentences(src_stcs)
        assert len(src_stcs) == len(pred_stcs)

        predicted_file = self.save_dir + f"/pred-{epoch}_{step}.txt"
        # 把原始句子的id，原始句子，人工校对过的结果，模型预测结果，
        # 预测结果是否完全正确 逐行保存到pred.txt中
        with open(predicted_file, "w", encoding="utf-8") as f:
            f.write("ID\t原句\t正确句\t预测句")
            f.write("\t原句与正确句相同\t原句与预测句相同\t预测句与正确句相同\n")
            for id, src, tgt, pred in zip(ids, src_stcs, tgt_stcs, pred_stcs):
                same_src_tgt = "YES" if src == tgt else "NO"
                same_src_pred = "YES" if src == pred else "NO"
                same_tgt_pred = "YES" if tgt == pred else "NO"
                f.write(f"{id}\t{src}\t{tgt}\t{pred}\t{same_src_tgt}")
                f.write(f"\t{same_src_pred}\t{same_tgt_pred}\n")

        return evaluate_file(predicted_file,
                             has_header=True, 
                             save_metrics=True)

    def _rm_tmp_files(self):
        '''
        删除path目录下不需要的预测结果文件
        '''
        path = Path(self.save_dir)
        # 提取出以pt结尾的文件名称，例如best_stc_1_13000.pt
        model_names = [f.name for f in path.glob('*.pt')]
        # 提取出文件名称中的epoch和step信息，如1_13000
        model_steps = [re.search(r'\d+_\d+', f).group() for f in model_names]
        accept_steps = set(model_steps)
        
        # 所有以pred-开始的文件，如果所在的step没有落在accept_steps中，则删除
        pred_files = [f for f in  path.glob('pred-*.*')]
        pred_steps = [re.search(r'\d+_\d+', f.name).group() \
            for f in pred_files]
        for f, step in zip(pred_files, pred_steps):
            if step not in accept_steps:
                L.debug(f'删除不需要的预测结果文件：{f.name}')
                f.unlink()
                
    def _save_config(self) -> None:
        """保存参数"""
        config = {
            "task_name": self.task_name,
            "train_path": self.train_path,
            "val_path": self.val_path,
            "test_path": self.test_path,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epoch": self.epoch_num,
            "model_name": self.model_name,
            "last_ckpt_file": self.last_ckpt_file,
            "dropout_rate": self.dropout_rate,
            "base_model": self.base_model,
            "correct_f1_weight": self.correct_f1_weight,
            "additional_confidence": self.additional_confidence,
            "additional_del_confidence": self.additional_del_confidence,
            "seed": self.seed,
            "cuda_num": self.cuda_num,
            "num_workers": self.num_workers,
            "amp": self.amp,
            "model_type": self.model_type,
            "label_loss_weight": self.label_loss_weight,
            "MAX_INPUT_LEN": conf.MAX_INPUT_LEN,
            "ENHANCE_SPECIAL_CHARS": conf.ENHANCE_SPECIAL_CHARS,
            "datetime": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        }
        config_file = f"{self.save_dir}/train_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(config, indent=4, ensure_ascii=False))