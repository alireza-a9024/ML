import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Dataset
df = pd.read_csv("dataset/Iris.csv")

# Preparing the data
x = df.drop(columns=["Species"])
y = df["Species"]

# Split data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)


# Predictions
y_pred = model.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
sns.pairplot(df, hue="Species")
plt.show