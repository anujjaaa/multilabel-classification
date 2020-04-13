import pandas as pd
import numpy as np

df= pd.read_csv('/home/sunbeam/PycharmProjects/medium/final.csv')
labels = []

# x=df.iloc[:,np.array([1,2])]
y=df.iloc[:,np.array([3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])]

x1=df.Title
x2=df.Body

from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(stop_words="english").fit(x1)
header=DataFrame(cv.transform(x1).todense(),columns=cv.get_feature_names())
cvArticle=cv.fit(x2)
article=DataFrame(cvArticle.transform(x2).todense(),columns=cvArticle.get_feature_names())

import pandas as pd
x9=pd.concat([header,article],axis=1)
x9 = x9.loc[:,~x9.columns.duplicated()]
# print(x9)

from sklearn.feature_extraction.text import TfidfTransformer
tfidfhead=TfidfTransformer().fit(x9)
x=DataFrame(tfidfhead.transform(x9).todense())
# tfidfart=TfidfTransformer().fit(article)
# art=DataFrame(tfidfart.transform(article).todense())
# print(x)

import pandas as pd
# x=pd.concat([head,art],axis=1)
# x = x.loc[:,~x.columns.duplicated()]
# print(x)

from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=7539562,test_size=0.2)


from sklearn.model_selection import train_test_split
# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=4599, test_size=0.2)
#45414,456,4595,4599,45998
# print(xtrain)
# print(xtest)
# print(type(xtest))
# xtrain = pandas.get_dummies(xtrain, columns=['Embarked'])


#=============================================================================================================

test1= pd.read_csv('/home/sunbeam/PycharmProjects/medium/test1.csv')

from sklearn.feature_extraction.text import CountVectorizer
z1=test1.Title
z2=test1.Body



from pandas import DataFrame

cv=CountVectorizer(stop_words="english").fit(z1)
header=DataFrame(cv.transform(z1).todense(),columns=cv.get_feature_names())
cvArticle=cv.fit(z2)
article=DataFrame(cvArticle.transform(z2).todense(),columns=cvArticle.get_feature_names())

import pandas as pd
z=pd.concat([header,article],axis=1)
z = z.loc[:,~z.columns.duplicated()]
# print(z)

result = pd.DataFrame().reindex_like(x9)
# print(result)
result=result.drop(result.index[0:2095])
# result=result.drop(result.iloc[:,45611:45974],inplace=True)
result = result.append(z, ignore_index=True, sort=False)
result=result.fillna(0)
# print(result)

# result = result.loc[:,~result.columns.duplicated()]
# result.to_csv(r'/home/sunbeam/PycharmProjects/medium/projectData/result.csv')
# print(result)

from sklearn.feature_extraction.text import TfidfTransformer
tfidfhead=TfidfTransformer().fit(result)
test=DataFrame(tfidfhead.transform(result).todense())
# print(test)
# print(type(test))
# test.to_csv(r'/home/sunbeam/PycharmProjects/medium/projectData/testing.csv')
# print(test)

# test= test.loc[:,~test.columns.duplicated()]

# tfidfart=TfidfTransformer().fit(article)
# art=DataFrame(tfidfart.transform(article).todense())
# print(test)
# z=pd.concat([head,art],axis=1)


from sklearn import tree
from sklearn.metrics import accuracy_score
# # train
decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(xtrain, ytrain)

# save the model
from sklearn.externals import joblib

joblib.dump(decision_tree, 'saved_model.pkl')

# Load the model from the file
# saved_DT = joblib.load('saved_model.pkl')

# Use the loaded model to make predictions
# saved_DT.predict(test)
# print(saved_DT)


# # predict
# prediction2 = decision_tree.predict(test)
# print(prediction2)

#--------------------------------------------------------------------------------------------
# using Label Powerset
# from skmultilearn.problem_transform import LabelPowerset
# from sklearn.naive_bayes import GaussianNB
# classifier = LabelPowerset(GaussianNB())
# classify=classifier.fit(xtrain, ytrain)
# # predict
# predicitions = classify.predict(xtest)

# predicitons = np.argmax(predictions, axis= -1)
# print("the prediction is : ",predicitions)

# predictions1 = classifier.predict(xtest)
# print(predictions1)

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(ytest.astype(float),predictions)
# print(acc)
# # tags = []
# for val in predictions:
#     tags.append(val)

# print(tags)