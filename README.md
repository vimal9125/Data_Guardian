1. Project Title

Data Guardian AI – Intelligent Data Auditing & Quality Assurance System

2. Project Description

Data Guardian AI is an advanced AI-powered data auditing tool that automatically analyzes datasets to identify missing values, anomalies, correlations, and inconsistencies.
It provides interactive visual dashboards, automated anomaly detection using Isolation Forest, and actionable insights for improving data quality.

The system aims to help data scientists, analysts, and organizations maintain clean and reliable datasets for better decision-making and model accuracy.

3. Project Objectives

🔍 Automate Data Auditing: Automatically detect missing values, outliers, and data irregularities.

📊 Visualize Data Insights: Provide interactive and easy-to-understand dashboards.

🤖 AI-Based Anomaly Detection: Use machine learning algorithms to identify data anomalies.

📈 Improve Data Quality: Suggest recommendations to clean and enhance dataset integrity.

🧠 Simplify Data Review: Enable users to upload any dataset and instantly get an AI summary.

4. Tools Used and Their Usage
Tool / Library	Purpose
Streamlit	For creating an interactive web-based dashboard.
Pandas	For data loading, cleaning, and analysis.
NumPy	For efficient numerical computations.
Plotly Express	For building professional and interactive data visualizations.
Scikit-learn (Isolation Forest, PCA, StandardScaler)	For anomaly detection, data scaling, and dimensionality reduction.
Datetime	For generating timestamps and reports.
5. Project Details

Below are screenshots representing the major sections and functionalities of Data Guardian AI:

📋 Dataset Overview

Displays basic dataset statistics including rows, columns, missing values, and correlation heatmap.
(Example Screenshot:)
🖼️ Shows top 10 records, data types, and a correlation matrix heatmap.

📉 Missing Value Analysis

Detects missing values per column and displays results with interactive bar charts.
(Example Screenshot:)
🖼️ Highlights columns with the highest percentage of missing data.

🚨 Anomaly Detection (Isolation Forest)

Identifies outliers in the dataset using the Isolation Forest algorithm.
Provides a pie chart of anomaly distribution and a 2D PCA scatter plot.
(Example Screenshot:)
🖼️ Shows anomalies in red and normal points in green.

🧠 AI Summary & Recommendations

Generates a health score based on missing values and anomalies.
Provides automated insights and actions for data improvement.
(Example Screenshot:)
🖼️ Shows dataset health progress bar and key suggestions.

6. Project Conclusion & Learnings
✅ Conclusion

Data Guardian AI successfully automates the process of data auditing and quality analysis.
It provides an AI-powered approach to identifying data issues and presenting them visually for better understanding.

🧩 What I Learned

Implementing machine learning algorithms (Isolation Forest, PCA) for real-world anomaly detection.

Building interactive dashboards with Streamlit and Plotly.

Applying data preprocessing and standardization using Scikit-learn.

Designing user-friendly interfaces and clean UI for data visualization.

Understanding data quality metrics and their impact on AI model performance.

📦 How to Run
pip install -r requirements.txt
streamlit run guardian_ai_v2.py
