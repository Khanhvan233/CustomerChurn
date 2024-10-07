from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,classification_report


#function to create a model using SMOTE
def prepare_model_smote(df,class_col,cols_to_exclude): 
#Synthetic Minority Oversampling Technique. Generates new instances from existing minority cases that you supply as input. 
  cols=df.select_dtypes(include=np.number).columns.tolist() 
  X=df[cols]
  X = X[X.columns.difference([class_col])]
  X = X[X.columns.difference(cols_to_exclude)]
  y=df[class_col] ##Selecting y as a column
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
  sm = SMOTE(random_state=0, sampling_strategy=1.0)
  X_train, y_train = sm.fit_resample(X_train, y_train) 
  return(X_train, X_test, y_train, y_test)


  
def run_model(X_train,X_test,y_train,y_test):
  ##Fitting the decision tree
  dectree = DecisionTreeClassifier(random_state = 13,criterion = 'entropy')
  dectree.fit(X_train, y_train)
  ##Predicting y values
  y_pred = dectree.predict(X_test)
  dectree_roc_auc = roc_auc_score(y_test, dectree.predict(X_test))
  print(classification_report(y_test, y_pred))
  print("The area under the curve is: %0.2f"%dectree_roc_auc)
  return (dectree,y_pred)