import time
import os
from pathlib import Path
import torch as t
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from transformers import AdamW
from tqdm import tqdm

from mectec import conf
from xiatian import L
from .dataset import FileSelectorDataset
from .collate import pad_collate
from .modeling import SelectorModel
from . import load_model

class SelectTrainer:
    def __init__(self,
                task_name:str,
                device,
                last_ckpt_file:str = None,
                start_epoch = 0,
                epoch_num=30,
                batch_size=32,
                learning_rate=1e-5,
                parallel=False,
                local_rank=-1,
                num_workers=4,
                cuda_num=1) -> None:
        self.task_name = task_name
        self.device = device
        self.save_dir = f"./model/{task_name}"
        self.start_epoch = start_epoch
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.parallel = parallel
        self.local_rank = local_rank
        self.num_workers = num_workers
        self.cuda_num = cuda_num
        
        conf.fix_seed(17) # 固定种子
        # 选择TensorBoard的输出位置
        self.writer = SummaryWriter(f'./logs/{task_name}') 
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        self.tokenizer = conf.bert_tokenizer
        self.model: SelectorModel = load_model(last_ckpt_file)
        self.model.to(self.device)
        if self.local_rank != -1:
            L.info(f"DDP加载模型:{self.local_rank}")
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[int(self.local_rank)],
                output_device=int(self.local_rank),
                find_unused_parameters=True)
        elif self.parallel:
            L.info(f"DP加载模型:{self.local_rank}")
            device_ids = list(range(self.cuda_num))
            self.model = t.nn.DataParallel(self.model, device_ids=device_ids)
        L.info(f"加载模型耗时：{time.time() - start_time}s")
        
        # 训练过程中的初始值
        self.best_model_names = [] # 已经保存的最好的模型列表
        self.accuracy_best = -1.0
    
    def load_data(self, train_path, val_path):
        train_dataset = FileSelectorDataset(train_path, self.tokenizer)
        val_dataset = FileSelectorDataset(val_path, self.tokenizer)

        if self.local_rank != -1:
            train_sampler = DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, 
                                          batch_size=self.batch_size, 
                                          sampler=train_sampler,
                                          num_workers=self.num_workers, 
                                          pin_memory=False, 
                                          prefetch_factor=2, 
                                          persistent_workers=True,
                                          collate_fn=pad_collate)
            val_dataloader = DataLoader(val_dataset, 
                                        shuffle=False, 
                                        batch_size=self.batch_size, 
                                        num_workers=self.num_workers,
                                        pin_memory=False, 
                                        prefetch_factor=2, 
                                        persistent_workers=True,
                                        collate_fn=pad_collate)
        else:
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, 
                                          sampler=train_sampler, 
                                          batch_size=self.batch_size, 
                                          pin_memory=True, 
                                          collate_fn=pad_collate)
            val_sampler = SequentialSampler(val_dataset)
            val_dataloader = DataLoader(val_dataset, 
                                        sampler=val_sampler, 
                                        batch_size=self.batch_size, 
                                        pin_memory=True, 
                                        collate_fn=pad_collate)
        return train_dataloader, val_dataloader
    
    def get_optimizer(self):
        # 参数组，每组参数可以指定自己的优化器参数，即每组参数可使用不同的优化策略
        param_optimizer = list(self.model.named_parameters())
        """实现L2正则化接口，对模型中的所有参数进行L2正则处理防止过拟合，包括权重w和偏置b"""
        # no_decay中存放不进行权重衰减的参数
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  
        optimizer_grouped_parameters = [
            {
                'params': [
                    p
                    for n, p in param_optimizer
                    if all(nd not in n for nd in no_decay)
                ],
                'weight_decay': 0.01,
            },
            {
                'params': [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0,
            },
        ]
        # 使用带有权重衰减功能的Adam优化器Adamw
        return AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
    
    def train(self, train_file, val_file, val_step=2000)->None:
        start_time = time.time()
        train_dataloader, val_dataloader = self.load_data(train_file, val_file)
        L.info(f"CUDA:{self.local_rank}加载耗时：{time.time() - start_time}s")

        optimizer = self.get_optimizer()
        # 学习率调整，如果在验证机上结果变化不大时，应该减小学习率
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1000)

        # 采用半精度
        scaler = t.cuda.amp.GradScaler()
        train_step_total,train_loss_total = 0, 0

        for epoch in range(self.start_epoch, self.epoch_num):
            if self.local_rank != -1: 
                train_dataloader.sampler.set_epoch(epoch)

            L.info("开始训练...")
            self.model.train()
            epoch_loss_total = 0
            loop = tqdm(train_dataloader, 
                        desc=f"Epoch:{epoch},local_rank:{self.local_rank}" )
            
            for step, batch_data in enumerate(loop,start=1):                    
                optimizer.zero_grad() # 梯度置零
                # 把数据移动到指定的device上
                token_ids, type_ids, mask, labels = map(
                    lambda tensor: tensor.to(self.device), batch_data)
                loss = self.model(token_ids, type_ids, mask, labels)

                train_step_total += 1

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                self.writer.add_scalar(
                    'lr', 
                    optimizer.state_dict()['param_groups'][0]['lr'], 
                    global_step=train_step_total)

                train_loss_total += loss.item()
                epoch_loss_total += loss.item()

                epoch_ave_loss = epoch_loss_total / step
                loop.set_postfix(loss=epoch_ave_loss)
                self.writer.add_scalar('loss', 
                                       epoch_ave_loss, 
                                       global_step=train_step_total)
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print( f"学习率：{lr}, loss:{epoch_ave_loss}")

                # 每隔val_step，则验证一次
                if train_step_total != 0 and train_step_total % val_step == 0:
                    L.info('验证并调整学习率...')
                    self.model.eval() # 进入测试模式，测试完后要打开训练模式
                    _, ave_val_loss = self.evaluate(val_dataloader)
                    # 根据验证集上的平均损失决定如何调整学习率
                    scheduler.step(ave_val_loss)
                    # 修改模式为训练模式
                    self.model.train()
            self.save_epoch(epoch, val_dataloader)        

    def save_epoch(self, epoch, val_dataloader) -> None:
        L.info('当前Epoch已完成，进入验证阶段...')
        self.model.eval() # 进入测试模式，测试完后要打开训练模式
        accuracy, _ = self.evaluate(val_dataloader)
        # 记录轮次的结果
        with open(os.path.join(self.save_dir, "val_epoch_result.txt"),
                  "a+",
                  encoding="utf-8") as f:
            f.write(f'epoch:{epoch}\t\t accuracy:{accuracy}\n')

        if accuracy <= self.accuracy_best: return None
        
        # 记录最佳结果并保存模型
        self.accuracy_best = accuracy
        model_name = os.path.join(self.save_dir, f"select_epoch_{epoch}.pt")
        L.info(f"保存DDP模型 CUDA:{self.local_rank}")
        model = self.model.module \
            if hasattr(self.model, "module") else self.model
        t.save(model.state_dict(), model_name, 
               _use_new_zipfile_serialization=False)

        # 如果保存的最佳模型超过了3个，则删除最旧的模型，最多保留3个
        self.best_model_names.append(model_name)
        if len(self.best_model_names) > 3: 
            os.remove(self.best_model_names.pop(0))

    def evaluate(self, dataloader):
        """对验证机进行评测，采用accuracy指标，返回精度"""        
        val_loss_total = 0
        correct_num = 0
        total_num = 0
        with t.no_grad():
            for batch_data in tqdm(dataloader, desc="Evaluating..."):
                token_ids, type_ids, mask, labels = \
                    map(lambda tensor: tensor.to(self.device), batch_data)
                pred_labels, _, loss = self.model(token_ids, type_ids, mask, 
                                                  labels, is_test=True)
                correct_num += t.sum(pred_labels == labels).item()
                total_num += len(labels)                
                val_loss_total += loss  
            accuracy = correct_num / total_num
            ave_val_loss = val_loss_total / len(dataloader)

        return accuracy, ave_val_loss
    