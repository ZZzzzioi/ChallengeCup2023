# -*- coding: utf-8 -*-
"""
Created on 2023.06.06
@author: ChallengeCup2023
"""

import json
import requests
import math
import time
import random
import pandas as pd
import numpy as np
import jieba
import re
import collections
from stylecloud import gen_stylecloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim import corpora
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator
import lda

'''
以“光伏骗局”为搜索关键词，按点击量排序，选取点击量超过10万的视频，抓取评论区文本，进行词频和聚类分析，绘制词云图等
'''

# 1.抓取评论文本，抓取字段包括用户名、发布时间、发布内容、点赞数
header = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}


def getAID(bvid):
    url = "http://api.bilibili.com/x/web-interface/view?bvid={bvid}".format(bvid=bvid)
    response = requests.get(url, headers=header)
    dirt = json.loads(response.text)
    aid = dirt['data']['aid']
    return aid


def getReplyPageNum(Aid):
    url="https://api.bilibili.com/x/v2/reply?&jsonp=jsonp&pn=1"+"&type=1&oid={}&sort=2".format(Aid)
    respond=requests.get(url)
    res_dirct=json.loads(respond.text)
    replyPageCount=int(res_dirct['data']['page']['count'])
    replyPageSize=int(res_dirct['data']['page']['size'])
    replyPageNum=math.ceil(replyPageCount/replyPageSize)
    return replyPageNum, replyPageSize


def getReplyContent(Aid, page):
    url = "https://api.bilibili.com/x/v2/reply/main?jsonp=jsonp&next={page}&type=1&oid={Aid}&mode=3".format(Aid=Aid, page=page)
    response = requests.get(url, headers=header)
    dirt = json.loads(response.text)
    ReplyData = dirt['data']['replies']
    return ReplyData


bvid_list = ['BV1hS4y1d78x', 'BV1mT411v7EL', 'BV1YF411a7An', 'BV1t7411b79y', 'BV1C94y1X7kV']
avid_list = [getAID(bvid) for bvid in bvid_list]


# 视频显示评论数包括了对评论的回复，因此实际所得评论少于视频下方显示的评论数
data_lists = list()
for avid in avid_list:
    replyPageNum, replyPageSize = getReplyPageNum(avid)
    for page in range(replyPageNum):
        replyData = getReplyContent(avid, page)
        time.sleep(random.randint(1, 3))
        for i in range(len(replyData)):
            username = replyData[i]['member']['uname']
            like = replyData[i]['like']
            content = replyData[i]['content']['message']
            ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(replyData[i]['ctime']))
            data_lists.append([username, like, content, ctime])
        print(avid, page)

name = ['username', 'like', 'content', 'ctime']
data_pd = pd.DataFrame(data=data_lists, columns=name)
data_pd.to_csv('./PublicOpinion.csv')
data_pd.to_excel('./PublicOpinion.xlsx')


# 2.对文本进行词频分析
jieba.load_userdict("./dict/SogouLabDic.txt")
jieba.load_userdict("./dict/dict_baidu_utf8.txt")
jieba.load_userdict("./dict/dict_pangu.txt")
jieba.load_userdict("./dict/dict_sougou_utf8.txt")
jieba.load_userdict("./dict/dict_tencent_utf8.txt")
StopWords = './dict/Stopword.txt'
AnalysisText = 'Comment.txt'

stopwords = {}.fromkeys([ line.rstrip() for line in open('./dict/Stopword.txt') ])

Data = pd.read_csv('PublicOpinion.csv')
CommentData = Data['content']
CommentFile = CommentData.to_csv('Comment.txt', index=False)
string_list = list(Data['content'])

with open(StopWords, 'r', encoding='UTF-8') as meaninglessFile:
    stopwords = set(meaninglessFile.read().split('\n'))
stopwords.add(' ')

object_list = list()
for string in string_list:
    pattern = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|"')
    string_re = re.sub(pattern, '', string)
    string_re = string_re.replace('\n', '')

    string_seg = jieba.cut(string_re, cut_all=False, HMM=True)

    string_seg_stopwords = [word for word in string_seg if word not in stopwords]

    object_list.append(string_seg_stopwords)

object_list_all = [j for i in object_list for j in i]

word_counts = collections.Counter(object_list_all)
#word_counts_top = word_counts.most_common(101)
word_counts_list = list(word_counts.items())

name = ['word', 'count']
word_counts_top_pd = pd.DataFrame(data=word_counts_list, columns=name)
#word_counts_top_pd = word_counts_top_pd.drop(46) # 删除 \xa0
word_counts_top_pd.to_csv('./WordCount.csv', index=False)


# 3.绘制词云图
gen_stylecloud(
    file_path='./WordCount.csv',
    size=600,
    font_path=r'./dict/Songti.ttc',
    output_name='wordcloud_solar.png',
    icon_name='fas fa-sun',
)


# 4.聚类分析
# 导出分词文本
object_cut = [' '.join(i) for i in object_list]
object_cut_pd = pd.DataFrame(object_cut)
object_cut_pd.to_csv('./Comment_LDA.txt', index=False, header=0)

corpus_cut = []  
for line in open('Comment_LDA.txt', 'r').readlines():  
    corpus_cut.append(line.strip())

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus_cut)

word = vectorizer.get_feature_names_out()

# 计算类别数量
dictionary = corpora.Dictionary(object_list)  # 构建词典
corpus = [dictionary.doc2bow(text) for text in object_list]

def coherence(num_topics):
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30,random_state = 1)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
    ldacm = CoherenceModel(model=ldamodel, texts=object_list, dictionary=dictionary, coherence='c_v')
    print(ldacm.get_coherence())
    return ldacm.get_coherence()


x = range(1,15)
y = [coherence(i) for i in x]

f, ax = plt.subplots(figsize=(7, 3))
patch = ax.patch
patch.set_color("white")
patch_f = f.patch
patch_f.set_color('white')

ax = plt.gca() # 获取当前的axes
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

plt.plot(x,y,'o-',color = 'k',label="CNN-RLSTM")
plt.xlabel('主题数目', color='k')
plt.ylabel('主题一致性值', color='k')

plt.rcParams['font.family'] = ['SimSong'] 
matplotlib.rcParams['axes.unicode_minus']=False

plt.title('主题一致性值变化情况', color='k', fontsize=13)
plt.tick_params(axis='x',colors='k')
plt.tick_params(axis='y',colors='k')
x_major_locator=MultipleLocator(1)
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
ax.set_yticks([0.3, 0.4, 0.5])
plt.show()

# 其中分类为1，2，3类时主题一致性均较高，为方便研究与描述，选择3类
model = lda.LDA(n_topics=3, n_iter=500, random_state=1)
model.fit(X)

#节选部分分类结果进行输出
doc_topic = model.doc_topic_
print("shape: {}".format(doc_topic.shape))  
for n in range(100):  
    topic_most_pr = doc_topic[n].argmax()  
    print(u"文档: {} 主题: {}".format(n,topic_most_pr)) 

word = vectorizer.get_feature_names_out()
topic_word = model.topic_word_  
for w in word:  
    print(w,end=" ")
print('')

n = 10
for i, topic_dist in enumerate(topic_word):    
    topic_words = np.array(word)[np.argsort(topic_dist)][:-(n+1):-1]    
    print(u'*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

print("shape: {}".format(topic_word.shape))  
print(topic_word[:, :10])  
for n in range(3):  
    sum_pr = sum(topic_word[n,:])  
    print("topic: {} sum: {}".format(n,  sum_pr)) 

f, ax= plt.subplots(10, 1, figsize=(10, 10), sharex=True)
for i, k in enumerate([0,1,2,3,4,5,6,7,8,9]):
    patch = ax[i].patch
    patch.set_color("white")
    patch_f = f.patch
    patch_f.set_color('white')

    ax[i].spines['right'].set_color('black')
    ax[i].spines['top'].set_color('black')
    ax[i].spines['left'].set_color('black')
    ax[i].spines['bottom'].set_color('black')

    ax[i].stem(doc_topic[k,:], linefmt='r-',  
               markerfmt='ro', basefmt='w-') 
    
    ax[i].tick_params(axis='x',colors='k')
    ax[i].tick_params(axis='y',colors='k')
    
    ax[i].set_xlim(-1, 3)      #三个主题
    ax[i].set_ylim(0, 1.0)     #权重0-1之间
    ax[i].set_ylabel("", color='k')  
    ax[i].set_title("文本 {}".format(k+1), color='k')
    
ax[4].set_xlabel("主题类别", color='k')
plt.tight_layout()
plt.savefig("result.png")
plt.show()

f, ax= plt.subplots(3, 1, figsize=(8,6), sharex=True) #三个主题
for i, k in enumerate([0, 1, 2]):
    patch = ax[i].patch
    patch.set_color("white")
    patch_f = f.patch
    patch_f.set_color('white')

    ax[i].spines['right'].set_color('black')
    ax[i].spines['top'].set_color('black')
    ax[i].spines['left'].set_color('black')
    ax[i].spines['bottom'].set_color('black')

    ax[i].stem(topic_word[k,:], linefmt='b-',
               markerfmt='bo', basefmt='w-')

    ax[i].tick_params(axis='x',colors='k')
    ax[i].tick_params(axis='y',colors='k')

    ax[i].set_xlim(-1, 6336)
    ax[i].set_ylim(0, 0.05)
    ax[i].set_ylabel("词频", color='k')
    ax[i].set_title("主题 {}".format(k+1), color='k')
ax[1].set_xlabel("特征词序号", color='k')
plt.tight_layout()
plt.savefig("result2.png")
plt.show() 

data = pd.read_csv('./PublicOpinion.csv', index_col='Unnamed: 0')
data['topic'] = [doc_topic[i].argmax() for i in range(len(data))]
data.to_csv('./PublicOpinion_topic.csv', index=False)
data.to_excel('./PublicOpinion_topic.xlsx', index=False)
