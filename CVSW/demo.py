import pandas as pd
import jieba
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics


# 划分正向情感和负向情感
def make_label(df):
    df["sentiment"] = df["star"].apply(lambda x: 1 if x > 3 else 0)


def chinese_word_cut(text):
    return " ".join(jieba.cut(text))


def get_custom_stopwords(file):
    with open(file) as f:
        stop_words = f.read()
    stopwords_list = stop_words.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list


if __name__ == '__main__':
    # 读入csv文件
    df = pd.read_csv('data.csv', encoding='gb18030')
    make_label(df)
    # 将特征和标签分开
    X = df[['comment']]
    y = df.sentiment
    # 将每一行评论分词
    X['cutted_comment'] = X.comment.apply(chinese_word_cut)
    # 将数据分为训练集和测试集
    # random_state=1在不同环境中随机数取值一致
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    # 哈工大停用词表
    stop_words_file = "stopwordsHIT.txt"
    stopwords = get_custom_stopwords(stop_words_file)
    # 对中文语句向量化
    max_df = 0.8  # 在超过这一比例的文档中出现的关键词(过于平凡)去除掉
    min_df = 3  # 在低于这一数量的文档中出现的关键词(过于独特)去除掉
    vector = CountVectorizer(max_df=max_df,
                             min_df=min_df,
                             token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                             stop_words=frozenset(stopwords))
    term_matrix = pd.DataFrame(vector.fit_transform(X_train.cutted_comment).toarray(),
                               columns=vector.get_feature_names())
    # 基于贝叶斯定理的朴素贝叶斯分类器
    nb = MultinomialNB()
    pipe = make_pipeline(vector, nb)
    # 使用交叉验证
    cross_val_score(pipe, X_train.cutted_comment, y_train, cv=5, scoring='accuracy').mean()
    # 把模型拟合
    pipe.fit(X_train.cutted_comment, y_train)
    y_pred = pipe.predict(X_test.cutted_comment)
    # 模型分类准确率
    print(metrics.accuracy_score(y_test, y_pred))
    # 混淆矩阵
    print(metrics.confusion_matrix(y_test, y_pred))
