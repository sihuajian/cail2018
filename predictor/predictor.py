from sklearn.externals import joblib
from predictor.tokenizers.norm_tokenizer import NormalTokenizer


class Predictor(object):
    def __init__(self):
        self.tfidf = joblib.load('predictor/model-over/tfidf.model')
        self.law = joblib.load('predictor/model-over/law-lp.model')
        self.law_ld = joblib.load('predictor/model-over/law-ld-k=5.model')
        self.accu = joblib.load('predictor/model-over/accu-lp.model')
        self.accu_ld = joblib.load('predictor/model-over/accu-ld-k=5.model')
        self.time = joblib.load('predictor/model-over/time.model')
        self.batch_size = 1024 * 10
        self.nor_cut = NormalTokenizer()
        # self.cut = thulac.thulac(seg_only=True)
        # self.stopwords_list = []

    def predict_law_svm(self, y, vec):
        result = []
        y_str = str(y)
        if y_str != '':
            for res in y_str.split('\n'):
                index1 = res.find(',')
                index2 = res.find(')')
                y_int = int(res[index1 + 1:index2].strip())
                result.append(y_int + 1)

        if len(result) == 0:
            y = self.law.predict(vec)
            y_str = str(y)
            if y_str != '':
                for res in y_str.split('\n'):
                    index1 = res.find(',')
                    index2 = res.find(')')
                    y_int = int(res[index1 + 1:index2].strip())
                    result.append(y_int + 1)

        return result

    def predict_accu_svm(self, y, vec):
        result = []
        y_str = str(y)
        if y_str != '':
            for res in y_str.split('\n'):
                index1 = res.find(',')
                index2 = res.find(')')
                y_int = int(res[index1 + 1:index2].strip())
                result.append(y_int + 1)

        if len(result) == 0:
            y = self.accu.predict(vec)
            y_str = str(y)
            if y_str != '':
                for res in y_str.split('\n'):
                    index1 = res.find(',')
                    index2 = res.find(')')
                    y_int = int(res[index1 + 1:index2].strip())
                    result.append(y_int + 1)

        return result

    def predict_time(self, vec):

        y = self.time.predict(vec)[0]

        # 返回每一个罪名区间的中位数
        if y == 0:
            return -2
        if y == 1:
            return -1
        if y == 2:
            return 120
        if y == 3:
            return 102
        if y == 4:
            return 72
        if y == 5:
            return 48
        if y == 6:
            return 30
        if y == 7:
            return 18
        else:
            return 6

    def predict(self, content):
        fact_temp = [self.nor_cut.tokenize(c) for c in content]
        vec = self.tfidf.transform(fact_temp)
        p1 = self.accu_ld.predict(vec)
        p2 = self.law_ld.predict(vec)
        # p3 = self.time.predict(vec)
        svm_p1 = [self.predict_accu_svm(p1[i], vec[i]) for i in range(len(content))]
        svm_p2 = [self.predict_law_svm(p2[i], vec[i]) for i in range(len(content))]

        ret = []
        for i in range(len(content)):
            ret.append({'accusation': svm_p1[i],
                        'articles': svm_p2[i],
                        'imprisonment': 0})
        return ret


if __name__ == '__main__':
    output_file = "output.txt"
    predict_target_file = "../data_valid_acc_nolabel-noblank.txt"
    pre = Predictor()
    try:
        fobj = open(output_file, 'w')
    except IOError:
        print('output file open error:' + output_file)
    else:
        f = open(predict_target_file, "r")
        for line in f:
            result = pre.predict(line)
            fobj.write(str(result))
    fobj.close()
