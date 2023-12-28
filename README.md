# MECTEC文本纠错模型

音形义增强的序列到编辑中文拼写纠错模型

# 安装需要的包

```shell
conda create --name megec python=3.9
conda activate megec

pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

在zsh环境下，如果提示conda activate不存在activate参数，执行：

```shell
conda init zsh
```

## 指定预训练语言模型

下载chinese-roberta-wwm-ext，放入到./pretrain这个默认目录中，以方便本地运行。

## 实验复现

### 下载训练数据

github有文件大小限制，训练文件和汉字嵌入结果需要单独下载。

请从ReaLiSe（https://github.com/DaDaMrX/ReaLiSe）提供的网址下载训练数据文件：

You can also directly download the processed data from** **[this](https://drive.google.com/drive/folders/1dC09i57lobL91lEbpebDuUBS0fGz-LAk) and put them in the** **`data` directory. The** **`data`directory would look like this:

从上面“this"链接中，下载里面的trainall.time2.pkl, 导出文本文件，存入到./data/train_realise.txt文件。处理代码方式可参考experiment.ipynb中的代码。

导出文件的每行格式示例：

A2-0011-1	你好！我是张爱文。	你好！我是张爱文。
A2-0023-1	下个星期，我跟我朋唷打算去法国玩儿。	下个星期，我跟我朋友打算去法国玩儿。

预处理的ViT汉字嵌入结果请访问网盘下载。

### 重新训练

 打开admin目录下的train_final.sh，根据GPU硬件配置调整参数CUDA_VISIBLE_DEVICES、nproc-per-node、num_workers、cuda_num参数，默认为4。

执行train_final.sh进行训练

### 下载训练过的模型

TODO: 上传百度网盘

### 预测

1. 修改http_server.py中的models字典信息，与实际情况保持一致。
2. 在vscode中安装REST Client插件，然后打开tests/request.http文件，点击里面的“设置参数”，注意json中的num_iterations为2和batch_size为32。
3. 在“预测”下面点击“send request”即可执行测试
4. 预测完毕后，会在当前文件夹下生成预测结果文件predicated.tsv
