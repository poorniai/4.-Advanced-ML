import pandas as pd
import RFE_Classification_Helper
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


def logistic(X_train,X_test,y_train,y_test):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    classifier,Accuracy,report,X_test,y_test,cm = RFE_Classification_Helper.cm_prediction(classifier,X_test,y_test)
    return classifier,Accuracy,report,X_test,y_test,cm

def svm_linear(X_train,X_test,y_train,y_test):
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train,y_train)
    classifier,Accuracy,report,X_test,y_test,cm=RFE_Classification_Helper.cm_prediction(classifier,X_test,y_test)
    return classifier,Accuracy,report,X_test,y_test,cm

def svm_NL(X_train,X_test,y_train,y_test):
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf',random_state=0)
    classifier.fit(X_train,y_train)
    return RFE_Classification_Helper.cm_prediction(classifier,X_test,y_test)

def Navie(X_train,X_test,y_train,y_test):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train,y_train)
    return RFE_Classification_Helper.cm_prediction(classifier,X_test,y_test)

def KNN(X_train,X_test,y_train,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5,metric='minkowski',p=2)
    classifier.fit(X_train,y_train)
    return RFE_Classification_Helper.cm_prediction(classifier,X_test,y_test)

def Decision(X_train,X_test,y_train,y_test):
    from sklearn.tree import DecisionTreeClassifier
    classifier=DecisionTreeClassifier(criterion = 'entropy',random_state=0)
    classifier.fit(X_train,y_train)
    return RFE_Classification_Helper.cm_prediction(classifier,X_test,y_test)

def Random(X_train,X_test,y_train,y_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state=0)
    classifier.fit(X_train,y_train)
    return RFE_Classification_Helper.cm_prediction(classifier,X_test,y_test)


def rfe_classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf): 
    
    rfedataframe=pd.DataFrame(index=['Logistic','SVC','Random','DecisionTree'],columns=['Logistic','SVMl','SVMnl',
                                                                                        'KNN','Navie','Decision','Random'])

    for number,idex in enumerate(rfedataframe.index):
        
        rfedataframe['Logistic'][idex]=acclog[number]       
        rfedataframe['SVMl'][idex]=accsvml[number]
        rfedataframe['SVMnl'][idex]=accsvmnl[number]
        rfedataframe['KNN'][idex]=accknn[number]
        rfedataframe['Navie'][idex]=accnav[number]
        rfedataframe['Decision'][idex]=accdes[number]
        rfedataframe['Random'][idex]=accrf[number]
    return rfedataframe