import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("ad_click_dataset_imputed.csv")
data.drop("id", inplace=True, axis=1)
data.drop("full_name", inplace=True, axis=1)
y = data["click"]
x = data.drop("click", axis=1)

# each column is a categorical feature, so map it to a number
label_encoders = {}
for column in x.columns:
    le = LabelEncoder()
    x[column] = le.fit_transform(x[column])
    label_encoders[column] = le

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, shuffle=True, random_state=0
)
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy DecisionTreeClassifier: {accuracy}")

# use decision forest classifier
classifier = RandomForestClassifier(random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy RandomForestClassifier: {accuracy}")
