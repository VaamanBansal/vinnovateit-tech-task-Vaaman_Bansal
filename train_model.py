import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
Data=pd.read_csv(r"C:\Users\Lenovo\Desktop\Vinnovate\Dataset.csv")
Data = Data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
Data['Extracurricular_Activities']=Data['Extracurricular_Activities'].map({'Yes': 1, 'No': 0})
Data['Gender']=Data['Gender'].map({"Female":0,"Male":1})
Data['Grade_Numeric'] = Data['Grade'].map( {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1, 'F': 0})
X=Data.drop(columns=['Grade','Grade_Numeric'])
Y=Data['Grade_Numeric']
x_train, x_test, y_train, y_test = train_test_split(X,Y,stratify=Y,test_size=0.2,random_state=42)
f1 = x_train.drop(columns=['Gender','Extracurricular_Activities'])
f2= x_test.drop(columns=['Gender','Extracurricular_Activities'])
scaler = StandardScaler()
scaler.fit(f1)
X_train_scaled = scaler.transform(f1)
X_test_scaled = scaler.transform(f2)
x_train_final = pd.DataFrame(X_train_scaled, columns=f1.columns, index=f1.index)
x_test_final = pd.DataFrame(X_test_scaled, columns=f2.columns, index=f2.index)
x_train_final[['Gender','Extracurricular_Activities']]=x_train[['Gender','Extracurricular_Activities']]
x_test_final[['Gender','Extracurricular_Activities']]=x_test[['Gender','Extracurricular_Activities']]
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train_final, y_train)
joblib.dump(clf, 'model/grade_predictor.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("Feature order used in training:")
print(list(x_train_final.columns))