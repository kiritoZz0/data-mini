from io import StringIO
import pydotplus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  tree
from sklearn import datasets
from  sklearn.datasets import load_wine
from sklearn.model_selection import  train_test_split
from  sklearn.preprocessing import  Imputer
import graphviz


imp = Imputer(missing_values='NaN',strategy='mean',axis=0,verbose=0,copy=True) ##处理缺失值用平均值代替

model_sample = pd.read_csv('D:\model.csv')    ##训练集
test_sample = pd.read_csv('D:\\test.csv')     ##测试集
model_sample.set_index('user_id',inplace=True)
test_sample.set_index('user_id',inplace=True)
data_y = model_sample.iloc[:,0].values          ##训练集.target
label = model_sample[['y']]
model_sample = model_sample.drop('y',axis=1)

head= list(model_sample.columns.values)       ## feature_names
#print(head)

data_x0 = model_sample.iloc[:,0:199].values
data_x = imp.fit_transform(data_x0)          ##训练集.data

test_x0 = test_sample.iloc[:,0:199].values
test_x = imp.fit_transform(test_x0)           ##测试集.data
#print(data_x)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_x,data_y,test_size=0.3)   ##分割训练集

#print(Xtrain)

##寻找最优剪枝参数
#max = 0;
#for i in range(10,30):
# for j in range(50,80):
#         for z in range(50, 80):
#           clf = tree.DecisionTreeClassifier(random_state=30,splitter="random",max_depth= 15,min_samples_leaf=j+1,min_samples_split=z+1)
#           clf = clf.fit(Xtrain, Ytrain)
#           score = clf.score(Xtest, Ytest) #返回预测的准确度
#           if(score>max):
#               max =score
#               #best_depth = i
#               best_leaf = j
#               best_spilt = z
#print(max)
#print(best_depth)
#print(best_leaf)
#print(best_spilt)




clf = tree.DecisionTreeClassifier(random_state=40,splitter="random",max_depth= 13,min_samples_leaf=59,min_samples_split=50)
#clf = clf.fit(Xtrain, Ytrain)     ##拿70%训练集来训练
clf = clf.fit(data_x, data_y)       ##拿所有训练集来训练
score = clf.score(Xtest, Ytest) #返回预测的准确度
#score = clf.score(data_x, data_y) #返回预测的准确度
print(score)

##绘制决策树
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,  # 绘制决策树
		      feature_names=head,
		      class_names=['class1','class2'],
		     filled=True, rounded=True,
	           )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("tree.pdf")

#predictedY=clf.predict(data_x)
predictedY=clf.predict(test_x)     ##测试测试集数据
print(predictedY) ##0-未逾期 1-逾期



#结果存入excel表格中
data = pd.DataFrame(predictedY)
writer = pd.ExcelWriter('result-m.xlsx')		# 写入Excel文件
data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()
writer.close()
# count = 0
# for i in range(len(predictedY)):
# 	if(data_y[i] == predictedY[i]):
# 		count = count + 1
# print(count*1.0/len(predictedY))
