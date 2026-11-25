import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE



def rfeFeature(indep_X,dep_Y,n):
    rfelist=[]
    log_model=LogisticRegression(solver='lbfgs')
    RF = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    DT = DecisionTreeClassifier(criterion='gini', max_features='sqrt', splitter='best', random_state=0)
    svc_model = SVC(kernel='linear',random_state=0)
    rfemodellist = [log_model,svc_model,RF,DT]
    for i in rfemodellist:
        print(i)
        log_rfe = RFE(estimator=i, n_features_to_select=n)
        log_fit = log_rfe.fit(indep_X,dep_Y)
        log_rfe_feature = log_fit.transform(indep_X)
        rfelist.append(log_rfe_feature)
    return rfelist

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



