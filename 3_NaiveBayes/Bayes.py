# -*- coding: utf-8 -*-
"""
location: xwbank
time:   2018-08-29
author: Lintao Cheng
案例：
    某社区有很多留言，为确保社区留言环境的健康，需要判断留言是否带有侮辱性？从而决定是否系统自动屏蔽。
算法使用的是朴素贝叶斯：
    1:词组 W=[word1,word2,...,wordn]
    2:p(侮辱类|W) = p(W|侮辱类)*p(侮辱类)/C    #C为常数
                 = p(word1|侮辱类)*...*p(wordn|侮辱类) * p(侮辱类)/C
     :p(非侮辱类|W) = p(W|非侮辱类)*p(非侮辱类)/C    #C为常数
                 = p(word1|非侮辱类)*...*p(wordn|非侮辱类) * p(非侮辱类)/C                
    3:为了判断W是不是侮辱类，需要判断  p(侮辱类|W)和p(非侮辱类|W)的大小来确定W属于侮辱类还是非侮辱类:

"""

#1：训练样本及标签
"""
函数说明:加载数据
Returns:
	postingList - 训练集
	classVec - 样本标签
"""  
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],        #训练样本
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                                       #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec



#2：将训练样本最后合成为一个向量
"""
函数说明:将样本汇总成内容不重复的词汇表
Parameters:
	dataSet - 训练样本
Returns:
	list(vocabSet) - 返回不重复的词条列表，也就是词汇表
"""   
def createVocabList(dataSet):
    vocabSet = set([])                       #创建一个空的不重复列表
    for document in dataSet:               
        vocabSet = vocabSet | set(document)  #取并集
    return list(vocabSet)


"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
生成一个和词汇表list(vocabSet)一样大小的0向量returnVec，样本的中的单词如果在词汇表中就将returnVec中相应位置变为1否则变为0
Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词集模型
"""
#3：得到文档矩阵
def setOfWords2Vec(vocabList, inputSet):           #(词汇库， 样本)
    returnVec = [0] * len(vocabList)               #创建一个其中所含元素都为0的向量
    for word in inputSet:                          #遍历每个样本
        if word in vocabList:                      #如果样本中的词存在于词汇库中，则置1
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec    






#4：训练分类器
"""
函数说明:朴素贝叶斯分类器训练函数
将训练集分为:
Parameters:
	trainMatrix   - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
	p0Vect - 侮辱类的条件概率数组
	p1Vect - 非侮辱类的条件概率数组
	pAbusive - 文档属于侮辱类的概率
"""
import numpy as np
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)						    #计算训练的文档数目
	numWords = len(trainMatrix[0])						    #计算每篇文档的词条数
	pAbusive = sum(trainCategory)/float(numTrainDocs)		    #文档属于侮辱类的概率
	p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)     #创建numpy.zeros数组,
	p0Denom = 0.0; p1Denom = 0.0                        	    #分母初始化为0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:							#统计属于侮辱类样本 对应的词汇表每个单词条件概率，即P(单词1|1),P(单词2|1),P(单词3|1)···
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:										     #统计属于非侮辱类的条件概率所需的数据，即P(单词1|1),P(单词2|1),P(单词3|1)···
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = p1Num/p1Denom									#相除      
	p0Vect = p0Num/p0Denom          
	return p0Vect,p1Vect,pAbusive							#返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

"""
函数说明:朴素贝叶斯分类器分类函数
Parameters:
    vec2Classify - 待分类的词条数组
    p0Vec - 侮辱类的条件概率数组
    p1Vec -非侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    0 - 属于非侮辱类
    1 - 属于侮辱类
    
该函数的问题： 
    应为样本量的关系 很容易导致p(单词i|侮辱类)、p(单词i|非侮辱类)这样的条件概率函数为0 这就导致了p1和p0为0 出现无法比较情况，针对这样的问题我们引入拉普拉斯平滑。   
"""

from functools import reduce
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1    			#reduce函数实现条件概率P(单词i|侮辱类)累乘
	p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)      #reduce函数实现条件概率P(单词i|非侮辱类)累乘
	print('p0:',p0)
	print('p1:',p1)
	if p1 > p0:
		return 1
	else: 
		return 0

#****************************************************************测试1
postingList, classVec = loadDataSet()        #返回样本数据及标签值
myVocabList = createVocabList(postingList)   #将训练样本整理成向量   
print('myVocabList:\n', myVocabList)
# 遍历每个样本 每个样本中的词在词汇库中就赋值1否则就赋值0
trainMat = []
for postinDoc in postingList:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    
p0V, p1V, pAb = trainNB0(trainMat, classVec)
print('侮辱类条件概率:\n', p0V)
print('非侮辱类条件概率:\n', p1V)
print('classVec:\n', classVec)
print('pAb:\n', pAb)
testEntry = ['love', 'my', 'dalmation']									#测试样本1
thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
classifyNB(thisDoc,p0V,p1V,pAb)


#**************************************引入拉普拉斯平滑及对数函数*******************************************;
#拉普拉斯平滑
import numpy as np
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                            #计算训练的文档数目
    numWords = len(trainMatrix[0])                            #计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs)        #文档属于侮辱类的概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)     #创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0; p1Denom = 2.0                             #分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                            #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                                                #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)                            #取对数，防止下溢出         
    p0Vect = np.log(p0Num/p0Denom)         
    return p0Vect,p1Vect,pAbusive                            #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
#log(A*B) = log(A) + log(B）
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)        #对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#****************************************************************测试2
postingList, classVec = loadDataSet()        #返回样本数据及标签值
myVocabList = createVocabList(postingList)   #将训练样本整理成向量   
print('myVocabList:\n', myVocabList)
# 遍历每个样本 每个样本中的词在词汇库中就赋值1否则就赋值0
trainMat = []
for postinDoc in postingList:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    
p0V, p1V, pAb = trainNB0(trainMat, classVec)
print('侮辱类条件概率:\n', p0V)
print('非侮辱类条件概率:\n', p1V)
print('classVec:\n', classVec)
print('pAb:\n', pAb)
testEntry = ['love', 'my', 'dalmation']									#测试样本1
thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
classifyNB(thisDoc,p0V,p1V,pAb)
