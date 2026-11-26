import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from statistics import mode


def pca(indep_X,n):
    from sklearn.decomposition import PCA
    sc = StandardScaler()
    X_scaled = sc.fit_transform(indep_X)
    pca = PCA(n_components=n)
    pca_features = pca.fit_transform(X_scaled)
    return pca_features

def lda(indep_X,dep_Y,n):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    sc = StandardScaler()
    X_scaled = sc.fit_transform(indep_X)
    lda = LDA(n_components = n)
    lda_features = lda.fit_transform(X_scaled,dep_Y)
    return lda_features

def kernel(indep_X,n):
    from sklearn.decomposition import KernelPCA
    sc = StandardScaler()
    X_scaled = sc.fit_transform(indep_X)
    kcpa = KernelPCA(n_components = 2,kernel='rbf')
    kcpa_features = kcpa.fit_transform(X_scaled)
    return kcpa_features

def r2_prediction(regressor,X_test,y_test):
     y_pred = regressor.predict(X_test)
     from sklearn.metrics import r2_score
     r2=r2_score(y_test,y_pred)
     return r2

def cm_prediction(classifier,X_test,y_test):
    from sklearn.metrics import confusion_matrix
    y_pred=classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    Accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    return classifier,Accuracy,report,X_test,y_test,cm
    

def standard_split(indep_X,dep_Y):
    X_train,X_test,y_train,y_test=train_test_split(indep_X,dep_Y,test_size=0.25,random_state=0)
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    return X_train,X_test,y_train,y_test




    






                                    

    