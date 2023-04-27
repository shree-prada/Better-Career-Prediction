import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("career_compute_dataset.csv")
print(np.shape(dataset))
dataset.head()

data = dataset.iloc[:49,:-1].values

label = dataset.iloc[:49,-1]
labelencoder = LabelEncoder()
df = dataset
label = df.iloc[:49,-1]
original=label.unique()
label=label.values
label2 = labelencoder.fit_transform(label)
y=pd.DataFrame(label2,columns=["ROLE"])
numeric=y["ROLE"].unique()
y1 = pd.DataFrame({'ROLE':original, 'Associated Number':numeric})
print(y1)
labelencoder = LabelEncoder()
label = labelencoder.fit_transform(label)
y=pd.DataFrame(label,columns=["role"])
X = pd.DataFrame(data,columns=['sslc','hsc','cgpa','school_type','no_of_miniprojects','no_of_projects',
                                'coresub_skill','aptitude_skill','problemsolving_skill','programming_skill','abstractthink_skill',
                                'design_skill','first_computer','first_program','lab_programs','ds_coding','technology_used',
                                'sympos_attend','sympos_won','extracurricular','learning_style','college_bench','clg_teachers_know','college_performence','college_skills'])



# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)#Decision tree
X_train2,X_test2,y_train2,y_test2=train_test_split(X,y,test_size=0.3,random_state=10)#XGBoost

def Dec_tree(X_train,y_train,X_test,y_test):
  from sklearn import tree
  clf = tree.DecisionTreeClassifier()
  clf = clf.fit(X_train, y_train)
  # Prediction
  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test,y_pred)
  print('Model accuracy score with Decision Tree', accuracy_score(y_test, y_pred)*100)
  return accuracy*100,clf

def xgboost(X_test,y_test,clf):
    xgb_y_pred  = clf.predict(X_test)
    xgb_accuracy = accuracy_score(y_test,xgb_y_pred)
    print("accuracy=",xgb_accuracy*100)
    import pickle
    pickle.dump(clf, open("career.pickle.dat", "wb"))
    loaded_model = pickle.load(open("career.pickle.dat", "rb"))
    x_new = ['2', '3', '2', '3', '1', '1', '1', '1', '2', '4', '2', '2', '2', '3', '4', '1', '1', '3', '1', '2', '2', '4','3', '4', '5']
    new_pred = loaded_model.predict([x_new])
    print(new_pred[0])
    filter=y1.loc[y1['Associated Number'] ==new_pred[0]].values.tolist()[0][0]
    print(type(filter),filter)
    print(y1['Associated Number']==new_pred[0])
    return xgb_accuracy*100

acc, clf = Dec_tree(X_train, y_train, X_test, y_test)
xgboost(X_test2, y_test2, clf)


