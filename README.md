# Data Cleaning & Preprocessing - Titanic Dataset

### What I Did
- **Imported** the dataset and explored basic statistics using `.info()`, `.describe()`, and `.isnull().sum()`.

- **Analyzed and filled missing values:**
  - **Age**: Most missing age values occurred in rows where both `SibSp` and `Parch` were 0, suggesting these were solo travelers. I filled these nulls with the median age of solo travelers. The remaining null values were filled using the overall median age.
  - **Embarked**: Had only 2 missing values, which were filled using the mode of the column.
  - **Cabin**: Dropped due to a high percentage of missing values, making it unreliable for analysis.

- **Processed Name column:**
  - Removed text within parentheses to clean alternate/maiden names.
  - Split the cleaned `Name` column into two new columns: `First Name` and `Last Name`.

- **Handled categorical variables:**
  - Applied one-hot encoding to `Sex` and `Embarked` columns.
  - Converted resulting boolean columns into integer (0/1).

- **Outlier detection and removal:**
  - Used boxplots and the IQR (Interquartile Range) method to detect outliers in the `Fare` column.
  - Removed only the outliers for Pclass 3, as high Fare values in Pclass 1 were verified to be valid.
  - Also confirmed the correlation between `Pclass` and `Fare` using the heatmap. 

- **Standardization:**
  - Plotted histograms for both `Fare` and `Age` to check distribution.
  - Confirmed non-Gaussian distribution using the Shapiro-Wilk test.
  - Standardized both `Age` and `Fare` using StandardScaler from Scikit-learn.

### Tools Used
- Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn

### Dataset
[Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
