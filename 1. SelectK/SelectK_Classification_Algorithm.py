import pandas as pd
import SelectKHelper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from statistics import mode

def logistic(X_train,X_test,y_train,y_test):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    classifier,Accuracy,report,X_test,y_test,cm = SelectKHelper.cm_prediction(classifier,X_test,y_test)
    return classifier,Accuracy,report,X_test,y_test,cm

def svm_linear(X_train,X_test,y_train,y_test):
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train,y_train)
    classifier,Accuracy,report,X_test,y_test,cm=SelectKHelper.cm_prediction(classifier,X_test,y_test)
    return classifier,Accuracy,report,X_test,y_test,cm

def svm_NL(X_train,X_test,y_train,y_test):
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf',random_state=0)
    classifier.fit(X_train,y_train)
    return SelectKHelper.cm_prediction(classifier,X_test,y_test)

def Navie(X_train,X_test,y_train,y_test):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train,y_train)
    return SelectKHelper.cm_prediction(classifier,X_test,y_test)

def KNN(X_train,X_test,y_train,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5,metric='minkowski',p=2)
    classifier.fit(X_train,y_train)
    return SelectKHelper.cm_prediction(classifier,X_test,y_test)

def Decision(X_train,X_test,y_train,y_test):
    from sklearn.tree import DecisionTreeClassifier
    classifier=DecisionTreeClassifier(criterion = 'entropy',random_state=0)
    classifier.fit(X_train,y_train)
    return SelectKHelper.cm_prediction(classifier,X_test,y_test)

def Random(X_train,X_test,y_train,y_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state=0)
    classifier.fit(X_train,y_train)
    return SelectKHelper.cm_prediction(classifier,X_test,y_test)

def Select_Classification(n,acclog,accsvmline,accsvmnl,navie,knn,decision,random):
    dataframe = pd.DataFrame(index=['ChiSquare'],columns=['K-Val','Logistic','SVM_Linear','SVM_NL','KNN','Navie','Decision','Random','Mode'])
    for number,idex in enumerate(dataframe.index):
         dataframe['K-Val'][idex]=n
         dataframe['Logistic'][idex]=acclog[number]  
         dataframe['SVM_Linear'][idex]=accsvmline[number]
         dataframe['SVM_NL'][idex]=accsvmnl[number]
         dataframe['KNN'][idex]=navie[number]
         dataframe['Navie'][idex]=knn[number]
         dataframe['Decision'][idex]=decision[number]
         dataframe['Random'][idex]=random[number]
         all_accuracy = (acclog + accsvmline + accsvmnl +navie + knn + decision + random)
         dataframe['Mode'][idex] = mode(all_accuracy)
         return dataframe
    