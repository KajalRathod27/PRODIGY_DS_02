import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset (train.csv)
df = pd.read_csv('Dataset/train.csv')

# Check for missing values and handle them
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# Convert categorical columns into numeric codes
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# EDA - Visualizations

# Survival Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Distribution')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=20, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Gender Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', data=df, palette='Set2')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Fare Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Fare'], kde=True, bins=30, color='green')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# Correlation Heatmap (only numeric columns)
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Survival Rate by Gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Sex', data=df, palette='Set2')
plt.title('Survival Rate by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Survival Rate by Pclass
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='Pclass', data=df, palette='Set2')
plt.title('Survival Rate by Pclass')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Survival Rate by Age Category
age_bins = [0, 12, 18, 30, 50, 100]
age_labels = ['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
df['AgeCategory'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='AgeCategory', data=df, palette='Set2')
plt.title('Survival Rate by Age Category')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Survival Rate by Family Size (SibSp + Parch)
df['FamilySize'] = df['SibSp'] + df['Parch']
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', hue='FamilySize', data=df, palette='Set2')
plt.title('Survival Rate by Family Size')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()
