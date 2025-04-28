
import pandas as pd
import matplotlib.pyplot as p;t
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load and preprocess the dataset
df = pd.read_csv("Depression Student Dataset.csv")
df.columns = [col.strip().replace("?", "").replace(" ", "_") for col in df.columns]
df.rename(columns={
    "Have_you_ever_had_suicidal_thoughts_": "Suicidal_Thoughts",
    "Family_History_of_Mental_Illness": "Family_History",
}, inplace=True)

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Suicidal_Thoughts'] = df['Suicidal_Thoughts'].map({'Yes': 1, 'No': 0})
df['Family_History'] = df['Family_History'].map({'Yes': 1, 'No': 0})
df['Depression'] = df['Depression'].map({'Yes': 1, 'No': 0})
df['Dietary_Habits'] = df['Dietary_Habits'].map({'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2})
df['Sleep_Duration'] = df['Sleep_Duration'].map({
    'Less than 5 hours': 4, '5-6 hours': 5.5, '7-8 hours': 7.5, 'More than 8 hours': 9
})

# Split the data
X = df.drop('Depression', axis=1)
y = df['Depression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 1. Identify key factors contributing to depression
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n1. Feature Importances:")
print(importances)

# 2. Correlation between academic performance and mental health
print("\n2. Correlation with Depression:")
correlation_matrix = df[['Academic_Pressure', 'Study_Satisfaction', 'Study_Hours', 'Depression']].corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation with Depression")
plt.show()

# 3. Predictive model evaluation
y_pred = model.predict(X_test)
print("\n3. Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# 4. Visualizations using matplotlib/seaborn
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Gender', hue='Depression')
plt.title("Depression by Gender")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(data=df, x='Age', hue='Depression', bins=10, multiple="stack")
plt.title("Depression by Age Group")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Depression', y='Study_Hours')
plt.title("Study Hours vs Depression")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='Depression', y='Sleep_Duration')
plt.title("Sleep Duration vs Depression")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Family_History', hue='Depression')
plt.title("Family History vs Depression")
plt.show()

# 5. Evaluate impact of support mechanisms
print("\n5. Impact of Support Mechanisms:")
impact = df.groupby(['Family_History', 'Suicidal_Thoughts'])['Depression'].mean().unstack()
print(impact)

# 6. Recommendations
print("\n6. Data-Driven Recommendations:")
recommendations = [
    "1. Increase awareness and access to counseling services.",
    "2. Promote healthy sleep and dietary routines among students.",
    "3. Offer financial aid and stress management workshops.",
    "4. Create peer support networks and mentoring systems.",
    "5. Tailor academic curriculums to improve study satisfaction."
]
for r in recommendations:
    print(r)
