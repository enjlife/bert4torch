# bert4torch

REFERENCE

[BERT-pytorch](https://github.com/codertimo/BERT-pytorch)

[transformers](https://github.com/huggingface/transformers)

[Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)

## 一些模型下载
### Hugging Face bert-base-chinese
模型可以通过两种方式下载:
`bert.from_pretrained`自动下载，参数设置如下：  
`bert = BertModel.from_pretrained('bert-base-chinese', **{'cache_dir': 'path_to_save'})`  
[huggingface](https://huggingface.co/bert-base-chinese/tree/main) 下载


### 哈工大整词bert
[chinese-bert-wwm pytorch版本下载](https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main)  
[Roberta-wwm-ext-base pytorch版本下载链接](https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main)  
当然，可以直接在[huggingface](https://huggingface.co/) 搜索

## 一些任务
### THUCNews新闻分类
原数据集共有14个类别，有些类被剔除例如时尚新闻，标题文本中有过多"组图"  
最终保留10个类别'财经', '房产', '股票', '教育', '科技', '社会', '时政', '体育', '游戏', '娱乐'。  

训练集：约75w，验证集1w，测试集1w 准确率94.9%

### 情感分类Sentiment
采用苏神 [必须要GPT3吗？不，BERT的MLM模型也能小样本学习](https://spaces.ac.cn/archives/7764/comment-page-1#comments) 的思路和数据。  

|     | _满意。 | _满意，| _喜欢。| _喜欢，| _理想。| _理想，|
|-----| ----- | ----- | ----- | ----- | ----- | ----- | 
|Bert-base-chinese| 69.64% | 73.66% |64.42% | 68.26% | 60.16% | 68.81% |
|Roberta-wwm-ext-base| 82.14% | 80.96% |74.34% | 76.44% | 74.99% | 78.98% |


