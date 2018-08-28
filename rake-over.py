from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from skmultilearn.ensemble import RakelD

import data_util
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from skmultilearn.problem_transform import LabelPowerset
from predictor.tokenizers.norm_tokenizer import normalize_cut_text
dim = 200000
stop_words = []


def train_tfidf(train_data):
    tfidf = TFIDF(
        min_df=5,
        max_features=dim,
        ngram_range=(1, 2),
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=True
    )
    tfidf.fit(train_data)

    return tfidf


def train_SVC_LP(vec, label):
    classifier = LabelPowerset(classifier=LinearSVC(), require_dense=[False, True])
    classifier.fit(vec, label)
    return classifier


def train_SVC(vec, label):
    SVC = LinearSVC()
    SVC.fit(vec, label)
    return SVC


def rakeld_ensemble(vec, label):
    problem_transform_classifier = LabelPowerset(classifier=LinearSVC(), require_dense=[False, True])
    classifier = RakelD(classifier=problem_transform_classifier, labelset_size=5)
    classifier.fit(vec, label)
    return classifier


if __name__ == '__main__':
    print('read accu and law')
    law, accu, lawname, accuname = data_util.init()

    # print('reading stopwords...')
    # stop_word = data_util.read_stopwrods()

    print('reading traindata...')
    # alltext, accu_label, law_label, time_label = data_util.read_trainData('../data/data_train.json')
    alltext, accu_label, law_label, time_label = data_util.read_trainData_by_dir('../data_oversample_ori')
    print(len(accu_label))

    acc_multihot = data_util.transform_multilabel(accu_label, accu)
    law_multihot = data_util.transform_multilabel(law_label, law)
    # print('cut text and remove stopwords...')
    train_data = normalize_cut_text(alltext)
    print(train_data[0])

    print('train tfidf...')
    tfidf = train_tfidf(train_data)

    vec = tfidf.transform(train_data)

    print('saving tfidf')
    joblib.dump(tfidf, 'predictor/model-0712/tfidf.model', compress=6)

    print('accu rakeld_ensemble')
    accu_d = rakeld_ensemble(vec, acc_multihot)
    joblib.dump(accu_d, 'predictor/model-0712/accu-ld-k=5.model', compress=5)

    print('law rakeld_ensemble')
    accu_d = rakeld_ensemble(vec, law_multihot)
    joblib.dump(accu_d, 'predictor/model-0712/law-ld-k=5.model', compress=5)

    print('accu svc lp')
    accu_lp = train_SVC_LP(vec, acc_multihot)
    joblib.dump(accu_lp, 'predictor/model-0712/accu-lp.model', compress=9)
    print('law svc lp')
    law_lp = train_SVC_LP(vec, law_multihot)
    joblib.dump(law_lp, 'predictor/model-0712/law-lp.model', compress=9)
    print('time svc SVC')
    time = train_SVC(vec, time_label)
    joblib.dump(time, 'predictor/model-0712/time.model', compress=3)

    print('finish')

