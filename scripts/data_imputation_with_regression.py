import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv('ad_click_dataset.csv') 

print("Original 'ad_click_dataset.csv' CSV Data: \n") 
print(data.shape)

df_old = data.copy()
df_copy = data[['click', 'id', 'full_name']].copy()
data.drop('click', inplace=True, axis=1)
data.drop('id', inplace=True, axis=1)
data.drop('full_name', inplace=True, axis=1)

label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    non_null_values = data[column][data[column].notnull()]
    le.fit(non_null_values)
    data[column][data[column].notnull()] = le.transform(non_null_values)
    label_encoders[column] = le

imp_mean = IterativeImputer(estimator=DecisionTreeClassifier(random_state=0))
data_np = imp_mean.fit_transform(data)

data = pd.DataFrame(data_np, columns=data.columns, dtype='int')
for column in data.columns:
    data[column] = label_encoders[column].inverse_transform(data[column])
data['click'] = df_copy['click']
data['id'] = df_copy['id']
data['full_name'] = df_copy['full_name']

data = data[df_old.columns]

print("\nImputed 'ad_click_dataset.csv' CSV Data: \n")
data.to_csv('ad_click_dataset_imputed.csv', index=False)
