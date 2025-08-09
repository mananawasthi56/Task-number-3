# Task-number-3
import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns import zipfile
from sklearn-model_selection import train_test_split from sklearn.preprocessing import LabelEncoder from sklearn. tree import DecisionTreeClassifier, plot_tree from sklearn.metrics import classification report, accuracy_score, confusion_m
# * Corrected: Read CSV from inside the ZIP
zip_path = "C: /Users/manan/DownLoads/bank+marketing/bank-additional.zip"
csv_filename = "bank-additional/bank-additional-full.csv".
with zipfile.ZipFile(zip_path) as z: with z.open(csv_filename) as f:
df = pd.read_csv(f, sep='; ')
# Display basic info
print( "Shape of dataset:"
", df.shape)
print( "Target variable value counts: In", df['y'] value_counts ())
# Encode categorical variables using Label Encoding
label encoders = 0}
for column in df.select_dtypes (include=[ 'object']).columns:
le = LabelEncoder ()
df [column] = le. fit_transform(df[column])
label_encoders [column] = le
# Split data into features (X) and target (y)
X = df. drop("y", axis=1)
= df. drop ("y", axis=1)
y = df[ "y"]
# Train-test split
X_train, X_test, y_train, _test = train_test_split(X, y, test_size=0.2, random_state=42, stri
# Train Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=5) # limit depth for clarity
clf.fit(X_train, y_train)
# Predict and evaluate
y_pred = clf-predict(X_test)
print("InClassification Report: In", classification_ report(y_test, y_pred))
print ("Accuracy Score:", accuracy_score(y_test, y_pred))
# Confusion matrix
cm = confusion_matrixy_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt. title "Confusion Matrix")
plt.xlabel ("Predicted")
plt-ylabel ("Actual")
plt. show()
# Visualize Decision Tree
plt. figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns,
class_names=label_encoders[ y'].classes_, filled=True;
plt. title("Decision Tree Visualization")
plt.show)
