import pandas as pd
import DimensionalityReductionHelper
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from statistics import mode

def logistic(X_train,X_test,y_train,y_test):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    classifier,Accuracy,report,X_test,y_test,cm = DimensionalityReductionHelper.cm_prediction(classifier,X_test,y_test)
    return classifier,Accuracy,report,X_test,y_test,cm

def svm_linear(X_train,X_test,y_train,y_test):
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train,y_train)
    classifier,Accuracy,report,X_test,y_test,cm=DimensionalityReductionHelper.cm_prediction(classifier,X_test,y_test)
    return classifier,Accuracy,report,X_test,y_test,cm

def svm_NL(X_train,X_test,y_train,y_test):
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf',random_state=0)
    classifier.fit(X_train,y_train)
    return DimensionalityReductionHelper.cm_prediction(classifier,X_test,y_test)

def Navie(X_train,X_test,y_train,y_test):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train,y_train)
    return DimensionalityReductionHelper.cm_prediction(classifier,X_test,y_test)

def KNN(X_train,X_test,y_train,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5,metric='minkowski',p=2)
    classifier.fit(X_train,y_train)
    return DimensionalityReductionHelper.cm_prediction(classifier,X_test,y_test)

def Decision(X_train,X_test,y_train,y_test):
    from sklearn.tree import DecisionTreeClassifier
    classifier=DecisionTreeClassifier(criterion = 'entropy',random_state=0)
    classifier.fit(X_train,y_train)
    return DimensionalityReductionHelper.cm_prediction(classifier,X_test,y_test)

def Random(X_train,X_test,y_train,y_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state=0)
    classifier.fit(X_train,y_train)
    return DimensionalityReductionHelper.cm_prediction(classifier,X_test,y_test)

def Select_Classification(n,name,acclog,accsvmline,accsvmnl,navie,knn,decision,random):
    dataframe = pd.DataFrame(columns=[name,'Logistic','SVM_Linear','SVM_NL','KNN','Navie','Decision','Random','Mode'])
    for number,idex in enumerate(range(len(acclog))):
         dataframe.loc[idex, name] = n
         dataframe.loc[idex,'Logistic'] = acclog[number]
         dataframe.loc[idex,'SVM_Linear'] = accsvmline[number]
         dataframe.loc[idex,'SVM_NL'] = accsvmnl[number]
         dataframe.loc[idex,'KNN'] = knn[number]
         dataframe.loc[idex,'Navie'] = navie[number]
         dataframe.loc[idex,'Decision'] = decision[number]
         dataframe.loc[idex,'Random'] = random[number]

         all_accuracy = (acclog + accsvmline + accsvmnl + navie + knn + decision + random)
         dataframe.loc[idex,'Mode'] = mode(all_accuracy)
         return dataframe
    