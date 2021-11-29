# bert4torch

## 模型下载
### Hugging Face bert-base-chinese
模型可以通过两种方式下载:  
- `bert.from_pretrained`自动下载，参数设置如下：  
`bert = BertModel.from_pretrained('bert-base-chinese', **{'cache_dir': 'path_to_save'})`  
- [HuggingFace](https://huggingface.co/) 搜索下载

### 常用预训练模型下载链接
| 来源 | 模型 | 模型参数 |下载地址 | 
| ---- | ---- | ---- | ---- |
|未知|bert-base-chinese pytorch版本 | 12-layer, 768-hidden, 12-heads, 110M parameters | https://huggingface.co/bert-base-chinese  |
|[HFL](https://github.com/ymcui/Chinese-BERT-wwm)| chinese-bert-wwm-ext pytorch版本 | 12-layer, 768-hidden, 12-heads, 110M parameters |https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main
|                                               |Roberta-wwm-ext-base pytorch版本 | 12-layer, 768-hidden, 12-heads, 110M parameters | https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main |
|[UER](https://github.com/dbiir/UER-py) | mixed_corpus_bert_basepytorch版本 需要使用reference的转换脚本转一下layer名称 | 12-layer, 768-hidden, 12-heads, 110M parameters | https://share.weiyun.com/5QOzPqq |


## 文本分类
### THUCNews新闻分类--bert

原数据集共有14个类别，有些类被剔除例如时尚新闻，标题文本中有过多"组图"  
最终保留10个类别'财经', '房产', '股票', '教育', '科技', '社会', '时政', '体育', '游戏', '娱乐'。  

训练集：约75w，验证集1w，测试集1w 准确率94.9%

### 情感分类--PET
采用苏神 [必须要GPT3吗？不，BERT的MLM模型也能小样本学习](https://spaces.ac.cn/archives/7764/comment-page-1#comments) 的思路和数据。  

|     | _满意。 | _满意，| _喜欢。| _喜欢，| _理想。| _理想，|
|-----| ----- | ----- | ----- | ----- | ----- | ----- | 
|Bert-base-chinese| 69.64% | 73.66% |64.42% | 68.26% | 60.16% | 68.81% |
|Roberta-wwm-ext-base| 82.14% | 80.96% |74.34% | 76.44% | 74.99% | 78.98% |

### NSP-BERT
思路来自[NSP-BERT](https://github.com/sunyilgdx/NSP-BERT/) ，复现了电商评论数据集eprstmt（zero-shot）的效果，测试集最高准确率约为86.8%。  

| 模型 | 准确率 | 
| ----| ---- |
| bert-base-chinese | 82% |
| chinese-bert-wwm-ext | 83% |
| uer-mixed_corpus_bert_base | 86.8% |


## CRF
### CNN+CRF实现分词
CRF计算loss，采用viterbi算法计算分词结果。语料来自 [Bakeoff 2005](http://sighan.cs.uchicago.edu/bakeoff2005/) \
torch实现：CRF/cnn_word_seg_torch.py 验证集准确率约91.54%。\
keras实现：CRF/cnn_word_seg.py 来自苏神的[简明条件随机场CRF介绍（附带纯Keras实现）](https://spaces.ac.cn/archives/5542)


## REFERENCE

[BERT-pytorch](https://github.com/codertimo/BERT-pytorch)  
[transformers](https://github.com/huggingface/transformers)  
[Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)
