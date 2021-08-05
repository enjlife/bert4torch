# bert4torch

reference

[BERT-pytorch](https://github.com/codertimo/BERT-pytorch)

[transformers](https://github.com/huggingface/transformers)

[Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)

bert-base-chinese模型可以通过两种方式下载
- `bert.from_pretrained`自动下载，参数设置如下：`bert = BertModel.from_pretrained('bert-base-chinese', **{'cache_dir': r'pretrained_model/bert-base-chinese'})`
- `https://huggingface.co/bert-base-chinese/tree/main` 下载

## THUCNews
剔除的类别有：时尚（标题文本中有过多"组图"）  
保留的类别`['财经', '房产', '股票', '教育', '科技', '社会', '时政', '体育', '游戏', '娱乐']`