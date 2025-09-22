# ML Data Preprocessing Pipeline

This repository provides a **data cleaning and preprocessing script** using Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.  
It prepares raw data for **Machine Learning** by handling missing values, encoding categories, scaling numerical features, and removing outliers.

---

## 🚀 Features
- Import dataset and explore summary statistics
- Handle missing values (mean for numerical, mode for categorical)
- Convert categorical features into numerical using **Label Encoding**
- Normalize/Standardize numerical features
- Visualize outliers with **boxplots**
- Remove outliers using the **IQR method**
- Save cleaned dataset into `cleaned_data.csv`

---

## 📂 Repository Structure
│── data_preprocessing.py   # main script
│── data.csv                # sample dataset (user can replace)
│── requirements.txt        # dependencies
│── README.md               # documentation
