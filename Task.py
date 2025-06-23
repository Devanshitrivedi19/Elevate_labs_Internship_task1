import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro,f_oneway
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Titanic-Dataset.csv')

print(df.info())
print(df.describe())
print(df.duplicated())
print(df.isnull().sum())

# Age Pattern: Most of the Age Rows that are Null have 0 SibSp and 0 Parch (Assumption is these travellers are Solo Travellers So Filling these rows with median Age of solo travellers age present)
df.loc[(df['SibSp'] == 0) & (df['Parch'] == 0) & (df['Age'].isnull()),'Age']=df.loc[(df['SibSp'] == 0) & (df['Parch'] == 0) & (df['Age'].notnull()), 'Age'].median()

# Filling Null Age of non solo travellers with the median of overall data
df.fillna({'Age': df['Age'].median()},inplace=True)

# Null count for Embarked is only 2 and filling it with the value Embarked Mode
df.fillna({'Embarked': df['Embarked'].mode()[0]}, inplace=True)

#Dropping the Cabin column As its Null count is very large
df.drop('Cabin', axis=1, inplace=True)

# Dropping Preferred name and seperating first name and last name
df['Name'] = df['Name'].str.replace(r'\([^)]*\)', '', regex=True).str.strip()
print(df['Name'])
First_Name, Last_Name = df['Name'].str.split(', ', expand=True)[1], df['Name'].str.split(', ', expand=True)[0]
df.insert(2,'First Name',First_Name)
df.insert(3,'Last Name',Last_Name)
df.drop('Name', axis=1, inplace=True)


df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
df['Sex_male'] = df['Sex_male'].astype(int)

for col in df.select_dtypes(include='bool').columns:
    df[col] = df[col].astype(int)

# print(df.groupby('Pclass')['Fare'].describe())

# Using box plot to identify the Outliers for Fare
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.boxplot(x='Fare', data=df[df['Pclass'] == 1])
plt.title("Fare Outliers in Pclass=1")

plt.subplot(1, 3, 2)
sns.boxplot(x='Fare', data=df[df['Pclass'] == 2])
plt.title("Fare Outliers in Pclass=2")

plt.subplot(1, 3, 3)
sns.boxplot(x='Fare', data=df[df['Pclass'] == 3])
plt.title("Fare Outliers in Pclass=3")
plt.tight_layout()
plt.show()

Q1 = df[df['Pclass'] == 3]['Fare'].quantile(0.25)
Q3 = df[df['Pclass'] == 3]['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[~((df['Pclass'] == 3) & (df['Fare'] > upper))]
# print(df.groupby('Pclass')['Fare'].describe())


# Compute correlation matrix
corr = df.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Heatmap")
plt.show()


plt.subplot(1, 2, 1)
sns.histplot(df['Fare'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df['Age'], kde=True, bins=30, color='salmon')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# As the Histograms shows the Bell Shaped Curve it is possible that it Gaussian (Normal) distribution So confirming it by calculating the p value

stat, p = shapiro(df['Fare'])
print(f"Shapiro-Wilk p-value: {p:.4f}") # Here the p value is 0.0000 Fare is not a Gaussian Distribution. 
stat, p = shapiro(df['Age'])
print(f"Shapiro-Wilk p-value: {p:.4f}") # Here the p value is 0.0000 Age is not a Gaussian Distribution. 

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print(df.info())
df.to_csv('Cleaned-Titanic-Dataset.csv', index=False)



