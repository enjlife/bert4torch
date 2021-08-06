# bert4torch

REFERENCE

[BERT-pytorch](https://github.com/codertimo/BERT-pytorch)

[transformers](https://github.com/huggingface/transformers)

[Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)

bert-base-chinese模型可以通过两种方式下载
- `bert.from_pretrained`自动下载，参数设置如下：  
`bert = BertModel.from_pretrained('bert-base-chinese', **{'cache_dir': 'path_to_save'})`
- [huggingface](https://huggingface.co/bert-base-chinese/tree/main) 下载

## THUCNews分类
原数据集共有14个类别，有些类被剔除例如时尚新闻，标题文本中有过多"组图"  
最终保留10个类别'财经', '房产', '股票', '教育', '科技', '社会', '时政', '体育', '游戏', '娱乐'。  

训练集：约75w，验证集1w，测试集1w 准确率94.9%

## 情感分类
采用苏神 [必须要GPT3吗？不，BERT的MLM模型也能小样本学习](https://spaces.ac.cn/archives/7764/comment-page-1#comments) 的思路和数据。  

| 前缀 | 准确率 |
| ----- | ----- |
| _满意。|69.64 |  
| _满意，|73.66%|

