import pandas as pd
import urllib3
from bs4 import BeautifulSoup

http=urllib3.PoolManager()


def spider(link):
    blogData = http.request('GET', link)
    soup = BeautifulSoup(blogData.data, 'html.parser')
    for links in soup.find_all('div', {'class': 'postArticle-readMore'}):
        link = links.find('a').get('href')
        CrawlAndFrame(link)

def CrawlAndFrame(link):
    print(link)
    blogData = http.request('GET', link)
    soup = BeautifulSoup(blogData.data, 'html.parser')
    article = ''
    tags = []
    heading = soup.find('h1').text
    for para in soup.find_all('p'):
        p = para.text
        # p=p.strip('\u')
        article = article + ' ' + p
    for mtags in soup.find_all('a', {'class': 'link u-baseColor--link'}):
        tags.append(mtags.text)
    # CreateDataFrame(list())
    someList = [heading, article, tuple(tags)]
    # print(someList[0])
    CreateDataFrame(someList)
    try:
        print(link)
        blogData = http.request('GET', link)
        soup = BeautifulSoup(blogData.data, 'html.parser')
        article = ''
        tags = []
        heading = soup.find('h1').text
        for para in soup.find_all('p'):
            p = para.text
        p = p.strip('/u')
        article = article + ' ' + p
        for mtags in soup.find_all('a', {'class ': 'link u - baseColor - link'}):
            tags.append(mtags.text)
            # CreateDataFrame(list())
            someList = [heading, article, tuple(tags)]
            # print(someList[0])
            CreateDataFrame(someList)
    except:
        pass

from pandas import DataFrame
column=['Title','Body']
dfBA=DataFrame(columns=column)
dfT=DataFrame(columns=[0,1,2,3,4])
content=[]
def CreateDataFrame(someList):
        t={}
        d={'Title':[someList[0]],'Body':[someList[1]]}
        for n in range(5):
            if len(someList[2])>n:
                t[n]=[someList[2][n]]
            else:
                t[n]=['0']
        toDf=DataFrame(data=d)
        global dfBA,dfT
        # print(dfBA)
        dfBA=dfBA.append(toDf)
        print(dfBA)
        # dfBA.to_pickle('sample1.pkl')
        dfT=dfT.append(DataFrame(data=t))
        # print(dfT)
        # dfT.to_pickle('sample1Tags.pkl')

        okList = []
        for cl in dfT.columns:
            for n in dfT[cl]:
                okList.append(n)
        okList = list(set(okList))
        del (okList[okList.index('0')])
        newDFT = DataFrame(columns=okList)
        for x in range(dfT.count()[0]):
            someDict = {}
            for d in okList:
                rowdata = list(dfT.iloc[x])
                if d in rowdata:
                    someDict[d] = 1
                else:
                    someDict[d] = 0
            newDFT = newDFT.append(someDict, ignore_index=True)
        newDFT.to_csv('blogTags.csv')

        # kList = []
        # for col in dfT.columns:
        #     for n in dfBA[col]:
        #         kList.append(n)
        # kList = list(kList)
        # del (kList[kList.index('0')])
        # newDF = DataFrame(columns=kList)
        # newDF = newDF.append(kList, ignore_index=True)
        # newDF.to_csv('blogData.csv')


        for column in dfBA.columns:
            for idx in dfBA[column].index:
                x = dfBA.get_value(idx, column)
                # try:
                #     x = x if type(x) == str else str(x).encode('utf-8', 'ignore').decode('utf-8', 'ignore')
                #     dfBA.set_value(idx, column, x)
                # except Exception:
                #     print('encoding error: {0} {1}'.format(idx, column))
                #     dfBA.set_value(idx, column, '')
                #     continue
        dfBA.to_csv('blogData.csv')



url=['https://medium.com/search?q=machine%20learning','https://medium.com/search?q=artificial%20intelligence',
     'https://medium.com/search?q=neural%20networks','https://medium.com/search?q=data%20science',
     'https://medium.com/search?q=big%20data','https://medium.com/search?q=deep%20learning']
for x in url:
    spider(x)
