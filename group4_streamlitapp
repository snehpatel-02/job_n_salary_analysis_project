import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Set page title
st.title("Job and Salary Analysis")

# Load the data
file_path = 'https://raw.githubusercontent.com/snehpatel-02/job_n_salary_analysis_project/refs/heads/main/Project_data.csv'
df = pd.read_csv(file_path)

# Renaming columns for simplicity
df.rename(columns={
    'Happiness_Salary': 'HappinessSalary',
    'Happiness_WorkLife': 'HappinessWorkLife',
    'Happiness_Learning': 'HappinessLearning',
    'Happiness_Mobility': 'HappinessMobility',
    'Job_Title': 'JobTitle',
    'Fav_Language': 'FavLanguage',
    'Career_Switch': 'CareerSwitch',
    'Difficulty_Breaking_In': 'DifficultyBreakingIn'
}, inplace=True)

# Display the dataframe head
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Display basic information
if st.checkbox('Show Data Info'):
    st.subheader("Data Info")
    st.text(df.info())

# Display summary statistics
if st.checkbox('Show Descriptive Statistics'):
    st.subheader("Descriptive Statistics")
    happiness_cols = ['HappinessSalary', 'HappinessWorkLife', 'Happiness_Coworkers',
                      'Happiness_Management', 'HappinessMobility', 'HappinessLearning']
    for col in happiness_cols:
        st.write(f"{col}:")
        st.write(f"Mean: {df[col].mean():.2f}")
        st.write(f"Median: {df[col].median():.2f}")
        st.write(f"Mode: {df[col].mode()[0]}")

# Group by Gender and show average happiness ratings
gender_stats = df.groupby('Gender')[happiness_cols].mean()
st.subheader("Gender-based Happiness Statistics")
st.write(gender_stats)

# Job Title analysis
top_jobs = df['JobTitle'].value_counts().nlargest(10).index
filtered_df = df[df['JobTitle'].isin(top_jobs)]

st.subheader("Top 10 Job Titles by Gender")
fig, ax = plt.subplots(figsize=(16, 7))
sns.countplot(data=filtered_df, y='JobTitle', hue='Gender', order=top_jobs, ax=ax)
ax.set_title('Top 10 Job Titles by Gender')
ax.set_xlabel('Count')
ax.set_ylabel('Job Title')
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', padding=2)
st.pyplot(fig)

# Plot Gender Distribution
st.subheader("Gender Distribution")
fig, ax = plt.subplots(figsize=(6, 6))
gender_counts = df['Gender'].value_counts()
ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
ax.set_title('Gender Distribution')
st.pyplot(fig)

# Top 3 Programming Languages by Gender
top_languages = df['FavLanguage'].value_counts().nlargest(3).index
top_lang_df = df[df['FavLanguage'].isin(top_languages)]

st.subheader("Top 3 Programming Languages by Gender")
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(data=top_lang_df, x='FavLanguage', hue='Gender', ax=ax)
ax.set_title('Top 3 Programming Languages by Gender')
ax.set_xlabel('Favorite Programming Language')
ax.set_ylabel('Count')
st.pyplot(fig)

# Interactive Plot for Job Titles
st.subheader("Top 15 Job Titles in the Dataset")
top_job_titles = df['JobTitle'].value_counts().nlargest(15).index
filtered_df = df[df['JobTitle'].isin(top_job_titles)]

fig, ax = plt.subplots(figsize=(16, 7))
sns.countplot(data=filtered_df, y='JobTitle', order=top_job_titles, palette='pastel', ax=ax)
ax.set_title('Top 15 Job Titles in the Dataset')
ax.set_xlabel('Count')
ax.set_ylabel('Job Title')
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', padding=2)
st.pyplot(fig)

# Bivariate Analysis - Happiness Salary by Gender
st.subheader("Happiness Salary by Gender")
fig, ax = plt.subplots()
sns.boxplot(data=df, x="Gender", y="HappinessSalary", ax=ax)
ax.set_title("Happiness Salary by Gender")
st.pyplot(fig)

# Scatter Plot - Work-Life vs Salary Happiness
st.subheader("Work-Life vs Salary Happiness by Gender")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="HappinessWorkLife", y="HappinessSalary", hue="Gender", ax=ax)
ax.set_title("Work-Life vs Salary Happiness by Gender")
st.pyplot(fig)

# Salary Parsing Function
def parse_salary(s):
    if pd.isna(s):
        return np.nan
    s = s.lower().replace(',', '').strip()
    if '-' in s:
        parts = s.split('-')
        try:
            low = parse_salary(parts[0])
            high = parse_salary(parts[1])
            return (low + high) / 2 if low and high else np.nan
        except:
            return np.nan
    if '+' in s:
        s = s.replace('+', '')
    match = re.match(r'(\d+(?:\.\d+)?)(k?)', s)
    if match:
        num, suffix = match.groups()
        num = float(num)
        if suffix == 'k':
            num *= 1000
        return num
    return np.nan

df['Salary_USD'] = df['Salary_USD'].apply(parse_salary)

# Plot Average Salary by Country and Gender
st.subheader("Average Salary by Country and Gender")
avg_salary = df.groupby(['Country', 'Gender'], as_index=False)['Salary_USD'].mean(numeric_only=True)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=avg_salary, x='Country', y='Salary_USD', hue='Gender', palette='Set2', ax=ax)
ax.set_title('Average Salary (USD) by Gender and Country')
ax.set_xlabel('Country')
ax.set_ylabel('Average Salary (USD)')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)
