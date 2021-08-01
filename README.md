# bert4torch

reference

[BERT-pytorch](https://github.com/codertimo/BERT-pytorch)

[transformers](https://github.com/huggingface/transformers)

[Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)

bert-base-chinese模型可以通过两种方式下载
- `bert.from_pretrained`自动下载，参数设置如下：`bert = BertModel.from_pretrained('bert-base-chinese', **{'cache_dir': r'pretrained_model/bert-base-chinese'})`
- `https://huggingface.co/bert-base-chinese/tree/main` 下载

## THUCNews
剔除的类别有：时尚（文本中过多"组图"）