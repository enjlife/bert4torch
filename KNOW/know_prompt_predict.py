import re
import six
import sys
import time
import requests
import json
import torch.nn
import torch.nn.functional as F
from tqdm import tqdm
sys.path.append('../')
from bert_torch import DatasetBase, BertTokenizer, BertForSequenceClassification, time_diff, \
                        sequence_padding, get_logger, BertConfig

logger = get_logger()


class DataIterator(DatasetBase):

    def __init__(self, data_list, batch_size, rand=False):
        super(DataIterator, self).__init__(data_list, batch_size, rand)

    def _to_tensor(self, datas):
        token_ids = [data[0] for data in datas]
        type_ids = [data[1] for data in datas]
        token_ids = sequence_padding(token_ids)
        type_ids = sequence_padding(type_ids)
        token_ids = torch.LongTensor(token_ids)
        type_ids = torch.LongTensor(type_ids)
        return token_ids, type_ids


class KNOWPredictor(object):
    patterns = ['学到很多！', '知识，技巧。', '学会啦！', '小知识。', '这是一篇知识或技巧的博客。', None]

    def __init__(self, model_path='', pretrained_path='../pretrained_model/bert-base-chinese/', batch_size=4,
                 max_len=180, p_index=1, device=0):
        self.model_path = model_path  # 加载的模型文件
        self.pretrained_path = pretrained_path  # 预训练配置文件
        self.num_classes = 2
        self.device = torch.device(device)  # gpu
        self.batch_size = batch_size
        self.max_len = max_len
        self.pattern = self.patterns[p_index]
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        self.model = self.init_model(pretrained_path, model_path)

    def init_model(self, pretrained_path, model_path):
        ws = torch.load(model_path, map_location='cpu')
        ws.pop("classifier2.weight", None)
        ws.pop("classifier2.bias", None)
        model = BertForSequenceClassification.from_pretrained(pretrained_path, self.num_classes, state_dict=ws)
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, ds):
        start_time = time.time()
        D, MIDS = self.load_dataset_predict(ds)
        data_iter = DataIterator(D, self.batch_size)
        logger.info('Load data time: {}'.format(time_diff(start_time)))
        start_time = time.time()
        scores = []
        with torch.no_grad():
            for (token_ids, type_ids) in tqdm(data_iter):
                token_ids, type_ids = token_ids.to(self.device), type_ids.to(self.device)
                logits = self.model(token_ids, type_ids)
                logits = F.softmax(logits, dim=-1)
                scores.extend(logits[:, 0].cpu().numpy().tolist())

        assert len(scores) == len(MIDS), 'Score num should equal mid num'
        logger.info('Predict time: {}, Scores num: {}, Mid num: {}'.format(time_diff(start_time), len(scores), len(mids)))

        return dict(zip(MIDS, scores))

    def load_dataset_predict(self, ds):
        logger.info('Load data using pattern: {}'.format(self.pattern))
        unused_list = ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]', '[unused6]', '[unused7]']
        PAD, CLS, MASK, SEP = '[PAD]', '[CLS]', '[MASK]', '[SEP]'
        D, MIDS = [], []
        if self.pattern:
            p_tokens = [CLS] + self.tokenizer.tokenize(self.pattern) + [SEP]
            p_type_ids = [0] * len(p_tokens)
        else:  # p-tuning
            p_tokens = [CLS] + unused_list + [SEP]
            p_type_ids = [0] * len(p_tokens)
        for d in ds:
            if not d:
                continue
            text = self.preprocess(d, max_len=self.max_len)
            mid = d['ID']
            if len(text) < 5:
                continue
            tokens = self.tokenizer.tokenize(text) + [SEP]
            type_ids = p_type_ids + [1] * len(tokens)
            tokens = p_tokens + tokens
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            D.append((token_ids, type_ids, 0))
            MIDS.append(str(mid))
        logger.info('Data num: {}, mid num: {}'.format(len(D), len(MIDS)))
        return D, MIDS

    @staticmethod
    def preprocess(d, max_len):
        text, content, l_title, v_title, v_info, v_voice = d.get('TEXT', ''), d.get('CONTENT', ''), \
                                                           d.get('LURL_TITLE', ''), d.get('VIDEO_TITLE', ''), eval(
            d.get('VIDEO_INFO', '{}')).get('display_name', ''), d.get('VIDEO_VOICE', '')
        if not text:
            text += content
        if l_title and l_title not in text:
            text += l_title
        if v_title and v_title not in text:
            text += v_title
        if v_info and v_info not in text:
            text += v_info
        if v_voice:
            text += v_voice[:100]
        if six.PY2:
            text = text.lower().decode()
        else:
            text = text.lower()
        text = re.sub(u'(https{0,1}://[a-zA-Z0-9/\.\-]*)|(\<.*?\>)|(@\S+)|(\[.*?\])|(我的评分：)', '', text)
        text = text.strip()
        text = re.sub(u'[\t\n\r ]', '，', text)
        text = ''.join(re.findall(u'[一-龥\w（）〈〉《》「」『』﹃﹄〔〕…—～﹏￥、【】，。？！：；“”‘]', text))
        text = re.sub(u'([？。！，])([？，。！]{1,})', '\\1', text)

        return text[:max_len]


def get_hbase_doc_field(mid, fields):
    url = u"http://getdata.search.weibo.com/getdata/querydata.php"
    i = 0
    while True:
        try:
            res_req = requests.get(url, params={u"mode": u"weibo", u"format": u"json", u"condition": mid})
            data_json = json.loads(res_req.text)
            break
        except Exception as e:
            if i == 2:
                print("mid: %s get failed" % mid, e)
                return None
            else:
                i += 1
                # continue
    if type(fields) == str:
        fields = [fields]
    ret = {f: data_json[f] for f in fields if f in data_json}
    return ret


if __name__ == '__main__':
    fields = ['ID', 'TEXT', 'CONTENT', 'LURL_TITLE', 'VIDEO_TITLE', 'VIDEO_INFO', 'VIDEO_VOICE']
    mids = open('../data/know/test0104.txt', 'r').readlines()
    mids = [line.split('\t')[0] for line in mids]
    ds = [get_hbase_doc_field(mid, fields) for mid in mids[:100]]
    predictor = KNOWPredictor(model_path='trained_prompt_2cls7_nsp.model', device=1)
    print(predictor.predict(ds))
