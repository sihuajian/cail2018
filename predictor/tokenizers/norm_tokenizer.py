from .tokenizer import Tokenizer
import re
import jieba
from multiprocessing.util import Finalize
from predictor import tokenizers
from multiprocessing import Pool as ProcessPool
from functools import partial

stopwords_path = 'predictor/tokenizers/stopwords.txt'
num_workers = 20


def init(tokenizer_class,str):
    global PROCESS_TOK
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


def normalize_single_cut(text):
    tokens = tokenize(text)
    return tokens


def normalize_cut_text(alltext):

    # nor = normalizer()
    # train_text = nor.seg_batch_text(alltext)
    tok_class = tokenizers.get_class('norm')
    train_text = []
    workers = ProcessPool(
        num_workers,
        initializer=init,
        initargs=(tok_class,'')
    )

    step = max(int(len(alltext) / 20), 1)
    batches = [alltext[i:i + step] for i in range(0, len(alltext), step)]
    _count = partial(normalize_single_cut)
    for i, batch in enumerate(batches):
        print('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for pa in workers.imap(_count, batch):
            train_text.append(pa)
            # print(pa)
    workers.close()
    workers.join()

    return train_text


class NormalTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        '''
        :param stopword_filepath: 停用词表路径
        :return:
        '''
        self.raw_json_list = []
        self.cut_list = []

        self.data_out = None
        self.stopword_filepath = stopwords_path
        self.pre_sub_pattern = [
            r'([\d一二三四五六七八九十零]+年|[\d一二三四五六七八九十零]+月|[\d一二三四五六七八九十零]+(日|号)|[\d一二三四五六七八九十零]+时|\d+分)+',
        ]
        self.pre_sub_replacer = [
            '日期',
        ]
        self.post_sub_pattern = [
            r'^(.*?)某$',
            r'^(.*?)某(\d+|甲|乙|丙|丁)$',
            r'^(.+)某(.+)$',
        ]
        self.post_sub_replacer = [
            '',
            '',
            '',
        ]
        # self.stopwordlist = None
        self.stopwordlist = self.read_data_to_list(self.stopword_filepath)

    @staticmethod
    def read_data_to_list(filepath):
        rslt_list = []

        with open(filepath, 'r', encoding='utf8') as fin:
            line = fin.readline()
            while line:
                line = line.strip()
                if line != "":
                    rslt_list.append(line)
                line = fin.readline()

        return rslt_list

    def tokenize(self, one_text):
        rslt_list = []
        text = one_text

        for i in range(len(self.pre_sub_pattern)):
            text = re.sub(self.pre_sub_pattern[i], self.pre_sub_replacer[i], text)
        tmp_cut_list = [word for word in jieba.lcut(text) if len(word) >= 2]
        for word in tmp_cut_list:
            if word in self.stopwordlist:
                continue
            if (re.match(r'^\d+(\.)?\d+$', word) != None):
                number = float(word)
                if (number < 1000 and number >= 0):
                    rslt_list.append('m0')
                elif 10000 > number >= 1000:
                    rslt_list.append('m' + str((int)(number / 1000)))
                elif (number < 100000 and number >= 10000):
                    rslt_list.append('mm' + str((int)(number / 10000)))
                elif (number < 1000000 and number >= 100000):
                    rslt_list.append('mmm' + str((int)(number / 100000)))
                elif (number < 10000000 and number >= 1000000):
                    rslt_list.append('mmmm' + str((int)(number / 1000000)))
                elif (number < 100000000 and number >= 10000000):
                    rslt_list.append('mmmmm' + str((int)(number / 10000000)))
                elif (number >= 100000000):
                    rslt_list.append('mmmmmm')
                continue
            elif (re.match(r'^[\d\.%a-zA-Z]+$', word) != None):
                continue
            elif (re.match(r'^(.*?)(县|市|乡|镇|州|村|区)$', word) != None):
                rslt_list.append('地区')
                continue
            for i in range(len(self.post_sub_pattern)):
                word = re.sub(self.post_sub_pattern[i], self.post_sub_replacer[i], word)
                if word == '':
                    break
            if word != '':
                rslt_list.append(word)

        return ' '.join(rslt_list)

    def seg_batch_text(self, text_list):
        rslt_batch_list = []

        n = 1
        for one_text in text_list:
            if n % 10000 == 0:
                print()
            rslt_batch_list.append(' '.join(self.seg_one_text(one_text)))
            n = n + 1
        return rslt_batch_list
