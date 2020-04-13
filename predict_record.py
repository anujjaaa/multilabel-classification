import pandas as pd
import numpy as np
df= pd.read_csv('/home/sunbeam/PycharmProjects/medium/final.csv')

def data(t1,t2):
    labels = []
    y = df.iloc[:, np.array(
        [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])]
    x1 = df.Title
    x2 = df.Body
    from sklearn.feature_extraction.text import CountVectorizer
    from pandas import DataFrame

    cv = CountVectorizer(stop_words="english").fit(x1)
    header = DataFrame(cv.transform(x1).todense(), columns=cv.get_feature_names())
    cvArticle = cv.fit(x2)
    article = DataFrame(cvArticle.transform(x2).todense(), columns=cvArticle.get_feature_names())
    import pandas as pd
    x9 = pd.concat([header, article], axis=1)
    x9 = x9.loc[:, ~x9.columns.duplicated()]

    import pandas as pd
    t1=pd.Series(t1)
    t2=pd.Series(t2)
    from sklearn.feature_extraction.text import CountVectorizer
    from pandas import DataFrame
    cv = CountVectorizer(stop_words="english").fit(t1)
    header = DataFrame(cv.transform(t1).todense(), columns=cv.get_feature_names())
    cvArticle = cv.fit(t2)
    article = DataFrame(cvArticle.transform(t2).todense(), columns=cvArticle.get_feature_names())
    import pandas as pd
    z = pd.concat([header, article], axis=1)
    z = z.loc[:, ~z.columns.duplicated()]
    result = pd.DataFrame().reindex_like(x9)
    result = result.drop(result.index[0:2095])
    result = result.append(z, ignore_index=True, sort=False)
    result = result.fillna(0)
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidfhead = TfidfTransformer().fit(result)
    test = DataFrame(tfidfhead.transform(result).todense())
    from sklearn import tree
    decision_tree = tree.DecisionTreeClassifier()
    # decision_tree = decision_tree.fit(xtrain, ytrain)
    from sklearn.externals import joblib
    # joblib.dump(decision_tree, 'saved_model.pkl')
    loaded_model=joblib.load('saved_model.pkl')
    prediction2 = loaded_model.predict(test)
    return prediction2