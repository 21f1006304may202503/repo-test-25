import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
metrics_dir = os.path.abspath(os.path.join(current_dir,".."))
model_dir  = metrics_dir

data_path = os.path.join(current_dir,"iris.csv")
metrics_path = os.path.join(metrics_dir,"metrics.csv")
model_path = os.path.join(model_dir,"model.joblib")

data = pd.read_csv(data_path)

train,test = train_test_split(data,test_size=0.4,stratify=data['species'],random_state=42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

params = {
  "max_depth":3,
  "random_state":1
}

mod_dt = DecisionTreeClassifier(**params)
mod_dt.fit(X_train,y_train)
y_pred = mod_dt.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
precision =  precision_score(y_test,y_pred,average='macro')
recall = recall_score(y_test,y_pred,average='macro')
f1 = f1_score(y_test,y_pred,average='macro')

metrics_df = pd.DataFrame({
  'Metric' : ['accuracy','precision','recall','f1'],
  'Score' : [accuracy,precision,recall,f1]
})
metrics_df.to_csv(metrics_path,index=False)
joblib.dump(mod_dt, model_path)