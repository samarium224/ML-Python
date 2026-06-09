# Python for Machine Guide
### NumPy · Pandas · Matplotlib

---

## Setup Checklist

Run this in a terminal or Jupyter Notebook:

```bash
pip install numpy pandas matplotlib jupyter
```

Then verify it works:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", plt.matplotlib.__version__)
```

**Recommended environment:** Jupyter Notebook or Google Colab (free, no install needed).

---

---

# MODULE 1: NumPy
### Estimated Time: 2.5–3 hours

---

## 1.1 — Why NumPy?

**Question:** *"Why can't we just use Python lists for math?"*

```python
import time

# Python list approach
py_list = list(range(1_000_000))
start = time.time()
result = [x * 2 for x in py_list]
print(f"Python list: {time.time() - start:.4f} seconds")

# NumPy approach
import numpy as np
np_array = np.arange(1_000_000)
start = time.time()
result = np_array * 2
print(f"NumPy array: {time.time() - start:.4f} seconds")
```

NumPy is 10–100x faster because it stores data in contiguous memory blocks and runs operations in optimized C code. In ML, you multiply millions of numbers constantly — speed matters enormously.

---

## 1.2 — Core Concepts

### Creating Arrays

```python
import numpy as np

# From a Python list
a = np.array([1, 2, 3, 4, 5])

# Useful shortcut creators
zeros   = np.zeros((3, 4))        # 3 rows, 4 cols of 0s
ones    = np.ones((2, 3))         # 2 rows, 3 cols of 1s
arange  = np.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)   # [0.0, 0.25, 0.5, 0.75, 1.0]
random  = np.random.rand(3, 3)    # 3x3 random floats between 0–1
```

**tip:** Try to answer: *"What shape do you expect this to be?"* before revealing with code.

### Shape and Dimensions

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.shape)   # (2, 3)  — 2 rows, 3 columns
print(a.ndim)    # 2       — number of dimensions
print(a.size)    # 6       — total number of elements
print(a.dtype)   # int64   — data type of elements
```

**ML connection:** In machine learning, data lives in 2D arrays — rows are samples, columns are features. An image is a 3D array (height × width × channels).

### Indexing and Slicing

```python
a = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

# Single element
print(a[0, 2])      # 30  — row 0, col 2

# Slicing rows and columns
print(a[1:, :2])    # rows 1+, first 2 cols → [[40, 50], [70, 80]]

# Boolean indexing (very important for ML!)
print(a[a > 50])    # [60, 70, 80, 90]
```

### Array Operations (Vectorization)

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

# Element-wise operations — no loops needed!
print(a + b)        # [11, 22, 33]
print(a * b)        # [10, 40, 90]
print(a ** 2)       # [1, 4, 9]
print(np.sqrt(a))   # [1.0, 1.414, 1.732]

# Summary statistics
print(a.sum())      # 6
print(a.mean())     # 2.0
print(a.max())      # 3
print(a.std())      # 0.816...
```

### Reshaping and Stacking

```python
a = np.arange(12)

# Reshape — critical in ML for feeding data into models
matrix = a.reshape(3, 4)    # 3 rows, 4 cols
flat    = matrix.flatten()  # Back to 1D

# Stacking arrays
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(np.vstack([x, y]))    # Stack vertically (row-wise)
print(np.hstack([x, y]))    # Stack horizontally (column-wise)
```

### Broadcasting

```python
# Broadcasting lets NumPy do math on arrays of different shapes
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

row    = np.array([10, 20, 30])

# NumPy "broadcasts" row across every row of matrix
print(matrix + row)
# [[11, 22, 33],
#  [14, 25, 36]]
```


## 1.3 — Exercises

**Exercise 1 (Easy):** Create a 4×4 identity-like matrix where the diagonal is `[1, 2, 3, 4]` instead of all 1s. (Hint: `np.diag`)

**Exercise 2 (Medium):** Given a 1D array of 100 random numbers between 0 and 1, replace all values less than 0.5 with 0 and all values ≥ 0.5 with 1. (This simulates a binary classifier's output threshold.)

**Exercise 3 (Medium):** Create a 5×5 array of random integers (1–100). Find the row with the highest sum. (Hint: `.sum(axis=1)` and `.argmax()`)

**Exercise 4 (Challenge):** Implement the formula for normalizing data (Min-Max Scaling) without any loops:
`normalized = (x - x.min()) / (x.max() - x.min())`
Apply it to a random array of 20 values. Verify the result is between 0 and 1.

---

## 1.4 — Key NumPy Concepts Summary

| Concept | Why It Matters for ML |
|---|---|
| Arrays vs. lists | Arrays are faster and support math operations |
| Shape & reshape | Data must be in the right shape for model inputs |
| Boolean indexing | Filter samples based on conditions (e.g., outliers) |
| Vectorized operations | Replace loops with fast batch computations |
| Broadcasting | Apply transformations across whole datasets |
| Statistics (mean, std) | Feature engineering and normalization |

---

---

# MODULE 2: Pandas
### Estimated Time: 3–4 hours

---

## 2.1 — Why Pandas?

What happens before you feed data into an ML model?

You clean it, explore it, and transform it. That's 70–80% of real ML work. Pandas is the tool for that.

**Analogy:** "Think of Pandas as Excel, but programmable, scalable to millions of rows, and infinitely more powerful."

---

## 2.2 — Core Concepts

### The Two Pandas Data Structures

```python
import pandas as pd

# Series — a single column with labels (index)
heights = pd.Series([170, 165, 182, 158], 
                    index=["Alice", "Bob", "Charlie", "Diana"])
print(heights["Bob"])     # 165
print(heights[heights > 165])  # Charlie

# DataFrame — a table (like a spreadsheet)
data = {
    "Name":   ["Alice", "Bob", "Charlie", "Diana"],
    "Age":    [25, 30, 22, 28],
    "Height": [170, 165, 182, 158],
    "Score":  [88, 92, 75, 95]
}
df = pd.DataFrame(data)
print(df)
```

### Loading Real Data

```python
# Load from CSV (most common in ML)
df = pd.read_csv("data.csv")

# First look at your data — always do this first!
print(df.head())          # First 5 rows
print(df.tail())          # Last 5 rows
print(df.shape)           # (rows, columns)
print(df.columns)         # Column names
print(df.dtypes)          # Data types per column
print(df.describe())      # Summary statistics
print(df.info())          # Non-null counts and dtypes
```

### Selecting Data

```python
# Select a single column → returns a Series
ages = df["Age"]

# Select multiple columns → returns a DataFrame
subset = df[["Name", "Score"]]

# Select rows by label (loc) or position (iloc)
df.loc[0]             # Row with index label 0
df.iloc[0]            # First row by position
df.loc[1:3, "Name":"Age"]   # Rows 1–3, columns Name to Age

# Filtering rows with conditions
high_scorers = df[df["Score"] > 90]
young_high   = df[(df["Age"] < 27) & (df["Score"] > 85)]
```

### Handling Missing Data

```python
# Check for missing values — this is always step 1 in real ML
print(df.isnull().sum())      # Count NaNs per column
print(df.isnull().mean())     # % missing per column

# Drop rows with ANY missing value
df_clean = df.dropna()

# Fill missing values (imputation — very common in ML)
df["Age"].fillna(df["Age"].mean(), inplace=True)       # Fill with mean
df["City"].fillna("Unknown", inplace=True)             # Fill with constant
df["Score"].fillna(df["Score"].median(), inplace=True) # Fill with median
```

### Adding and Modifying Columns

```python
# Add a new computed column
df["Height_m"] = df["Height"] / 100    # cm to meters
df["BMI"]      = df["Weight"] / df["Height_m"] ** 2

# Apply a function to a column
df["Grade"] = df["Score"].apply(lambda x: "Pass" if x >= 75 else "Fail")

# Map categories to numbers (encoding — essential for ML!)
df["Gender_code"] = df["Gender"].map({"Male": 0, "Female": 1})
```

### Grouping and Aggregating

```python
# Group by a category and get statistics
print(df.groupby("Grade")["Score"].mean())
print(df.groupby("City")["Age"].agg(["min", "max", "mean"]))

# Value counts — great for understanding class distributions
print(df["Grade"].value_counts())
print(df["Grade"].value_counts(normalize=True))  # As percentages
```

### Sorting and Ranking

```python
# Sort by one or more columns
df_sorted = df.sort_values("Score", ascending=False)
df_sorted = df.sort_values(["Grade", "Score"], ascending=[True, False])

# Rank values
df["Rank"] = df["Score"].rank(ascending=False)
```

### Combining DataFrames

```python
# Merge — like SQL JOIN
merged = pd.merge(df1, df2, on="ID", how="left")

# Concatenate rows
combined = pd.concat([df_train, df_test], ignore_index=True)
```

---

## 2.3 — ML-Specific Pandas Patterns

These are patterns you will use repeatedly in real ML projects:

```python
# 1. Separate features (X) from target (y)
X = df.drop("target_column", axis=1)
y = df["target_column"]

# 2. One-Hot Encoding categorical columns
df_encoded = pd.get_dummies(df, columns=["City", "Gender"])

# 3. Train/Test split using Pandas
train = df.sample(frac=0.8, random_state=42)
test  = df.drop(train.index)

# 4. Check class balance (vital for classification!)
print(y.value_counts(normalize=True))

# 5. Correlation matrix (to spot which features matter)
print(df.corr())
```

---

## 2.4 — Exercises

**Exercise 1 (Easy):** Load the Titanic CSV. How many passengers survived? What percentage is that?

**Exercise 2 (Medium):** Find the average age of survivors vs. non-survivors. Did younger passengers survive more often?

**Exercise 3 (Medium):** The "Age" column in Titanic has missing values. Fill them with the median age per passenger class (Pclass). (Hint: `groupby` + `transform`)

**Exercise 4 (Challenge):** Create a new column called `FamilySize = SibSp + Parch + 1`. Then group passengers into "Alone", "Small Family" (2–4), and "Large Family" (5+). Which group had the best survival rate?

---

## 2.5 — Key Pandas Concepts Summary

| Concept | Why It Matters for ML |
|---|---|
| DataFrame / Series | The standard container for ML datasets |
| `head()`, `describe()`, `info()` | First-look exploration before modeling |
| Filtering with conditions | Create subsets, isolate outliers |
| Handling nulls | Most real datasets are incomplete |
| `apply()` / `map()` | Feature transformation and encoding |
| `groupby()` | Understand patterns across categories |
| `get_dummies()` | Convert categories to numeric for models |

---

---

# MODULE 3: Matplotlib
### Estimated Time: 2.5–3 hours

---

## 3.1 — Why Matplotlib?

You can't understand data you can't see. Before any model, you must visualize.

Good visualizations help you:
- Spot outliers before they poison your model
- Understand feature distributions
- See relationships between variables
- Check if your model's predictions make sense

---

### The Anatomy of a Matplotlib Figure
```
Figure  → The whole canvas (like a page)
  └── Axes → The actual plot area (can have multiple per figure)
        ├── x-axis, y-axis
        ├── Title, labels
        └── The actual plotted data (lines, bars, dots, etc.)
```

**Always use this pattern** (teach the object-oriented style from day 1):

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()      # Create figure and axes
ax.plot([1, 2, 3], [4, 5, 6]) # Draw something
ax.set_title("My Plot")        # Add title
ax.set_xlabel("X Axis")        # Label x
ax.set_ylabel("Y Axis")        # Label y
plt.show()                     # Display
```

### Line Plot — Trends Over Time

```python
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 21)
train_loss = 1 / (1 + 0.3 * epochs) + np.random.rand(20) * 0.05
val_loss   = 1 / (1 + 0.25 * epochs) + np.random.rand(20) * 0.08

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, train_loss, label="Training Loss",   color="blue",   linewidth=2)
ax.plot(epochs, val_loss,   label="Validation Loss", color="orange", linewidth=2, linestyle="--")
ax.set_title("Model Training Progress")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Scatter Plot — Relationships Between Features

```python
fig, ax = plt.subplots()

# Simulate two features with some relationship
x = np.random.randn(100) * 10 + 50   # e.g., study hours
y = x * 0.8 + np.random.randn(100) * 5 # e.g., exam score

ax.scatter(x, y, color="steelblue", alpha=0.6, edgecolors="white", s=60)
ax.set_title("Study Hours vs. Exam Score")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Exam Score")
plt.show()
```

### Histogram — Feature Distributions

```python
ages = np.random.normal(loc=35, scale=10, size=500)

fig, ax = plt.subplots()
ax.hist(ages, bins=30, color="coral", edgecolor="white", alpha=0.8)
ax.set_title("Age Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
plt.show()
```

### Bar Chart — Comparing Categories

```python
labels  = ["Class A", "Class B", "Class C"]
counts  = [120, 85, 200]
colors  = ["#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots()
bars = ax.bar(labels, counts, color=colors, edgecolor="white", width=0.5)
ax.bar_label(bars, padding=3)   # Add value labels on bars
ax.set_title("Sample Count by Class")
ax.set_ylabel("Number of Samples")
ax.set_ylim(0, 240)
plt.show()
```

### Multiple Subplots — The Data Overview Dashboard

```python
import pandas as pd

# Simulated feature data
np.random.seed(42)
df = pd.DataFrame({
    "Age":    np.random.normal(35, 10, 300),
    "Income": np.random.exponential(40000, 300),
    "Score":  np.random.uniform(50, 100, 300),
    "Class":  np.random.choice([0, 1], 300)
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Feature Overview", fontsize=16)

axes[0, 0].hist(df["Age"],    bins=20, color="steelblue")
axes[0, 0].set_title("Age Distribution")

axes[0, 1].hist(df["Income"], bins=20, color="coral")
axes[0, 1].set_title("Income Distribution")

axes[1, 0].scatter(df["Age"], df["Score"], c=df["Class"], cmap="RdYlGn", alpha=0.5)
axes[1, 0].set_title("Age vs Score (colored by Class)")

axes[1, 1].boxplot([df[df["Class"]==0]["Income"], df[df["Class"]==1]["Income"]])
axes[1, 1].set_xticklabels(["Class 0", "Class 1"])
axes[1, 1].set_title("Income by Class")

plt.tight_layout()
plt.show()
```

### Heatmap — Correlation Matrix

```python
import pandas as pd

corr_matrix = df[["Age", "Income", "Score"]].corr()

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(im)

ax.set_xticks(range(len(corr_matrix)))
ax.set_yticks(range(len(corr_matrix)))
ax.set_xticklabels(corr_matrix.columns)
ax.set_yticklabels(corr_matrix.columns)

# Add value labels inside each cell
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                ha="center", va="center", fontsize=10)

ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
```

---

## 3.3 — Exercises for Students

**Exercise 1 (Easy):** Plot two lines on the same graph: `y = x²` and `y = 2x + 1` for x from -10 to 10. Add a legend, title, and grid.

**Exercise 2 (Medium):** Using the Titanic dataset, create a bar chart showing the survival count (0 vs 1) for each passenger class (1st, 2nd, 3rd).

**Exercise 3 (Medium):** Plot the age distribution for survivors and non-survivors on the same histogram using transparency (`alpha`). What do you notice?

**Exercise 4 (Challenge):** Build a 2×2 subplot figure for the Titanic dataset showing: (a) age histogram, (b) fare distribution, (c) survival by gender bar chart, (d) scatter of age vs fare colored by survival. Add a main title.

---

## 3.4 — Key Matplotlib Concepts Summary

| Plot Type | When to Use in ML |
|---|---|
| Line plot | Training curves (loss, accuracy over epochs) |
| Scatter plot | Visualize relationships between two features |
| Histogram | See how a feature is distributed |
| Bar chart | Compare class counts, category averages |
| Box plot | Spot outliers, compare distributions |
| Heatmap | Correlation matrix, confusion matrix |
| Subplots | Data overview dashboards |

---

---

# MODULE 4: Mini-Project — Putting It All Together
### Estimated Time: 2–3 hours

---

## The Project: Exploratory Data Analysis on the Titanic Dataset

This project uses all three libraries together, mirroring a real ML workflow.

### Step 1 — Load and Inspect (NumPy + Pandas)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df.shape)
print(df.head())
print(df.isnull().sum())
```

### Step 2 — Clean the Data (Pandas)

```python
# Fill missing ages with median age per class
df["Age"] = df.groupby("Pclass")["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Fill missing Embarked with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Drop columns not useful for analysis
df.drop(columns=["Cabin", "Ticket", "Name"], inplace=True)

print(df.isnull().sum())  # Should be 0
```

### Step 3 — Feature Engineering (NumPy + Pandas)

```python
# Create family size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Create age groups using NumPy
df["AgeGroup"] = pd.cut(df["Age"], 
                         bins=[0, 12, 18, 35, 60, 100],
                         labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])

# Encode gender
df["Sex_code"] = df["Sex"].map({"male": 0, "female": 1})

# Normalize fare using NumPy
fare_arr = df["Fare"].values
df["Fare_norm"] = (fare_arr - fare_arr.min()) / (fare_arr.max() - fare_arr.min())
```

### Step 4 — Explore and Visualize (Matplotlib + Pandas)

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Titanic EDA — Survival Analysis", fontsize=16, fontweight="bold")

# Plot 1: Overall survival
counts = df["Survived"].value_counts()
axes[0, 0].bar(["Did Not Survive", "Survived"], counts.values, color=["#E74C3C", "#2ECC71"])
axes[0, 0].set_title("Overall Survival")
axes[0, 0].set_ylabel("Count")

# Plot 2: Survival by class
survival_by_class = df.groupby("Pclass")["Survived"].mean()
axes[0, 1].bar(["1st Class", "2nd Class", "3rd Class"], 
               survival_by_class.values, color=["gold", "silver", "#CD7F32"])
axes[0, 1].set_title("Survival Rate by Class")
axes[0, 1].set_ylabel("Survival Rate")

# Plot 3: Survival by gender
survival_by_sex = df.groupby("Sex")["Survived"].mean()
axes[0, 2].bar(survival_by_sex.index, survival_by_sex.values, color=["#3498DB", "#E91E8C"])
axes[0, 2].set_title("Survival Rate by Gender")
axes[0, 2].set_ylabel("Survival Rate")

# Plot 4: Age distribution by survival
axes[1, 0].hist(df[df["Survived"]==0]["Age"].dropna(), bins=25, alpha=0.6, 
                color="#E74C3C", label="Did Not Survive")
axes[1, 0].hist(df[df["Survived"]==1]["Age"].dropna(), bins=25, alpha=0.6, 
                color="#2ECC71", label="Survived")
axes[1, 0].set_title("Age Distribution by Survival")
axes[1, 0].set_xlabel("Age")
axes[1, 0].legend()

# Plot 5: Family size vs survival
family_survival = df.groupby("FamilySize")["Survived"].mean()
axes[1, 1].plot(family_survival.index, family_survival.values, 
                marker="o", color="purple", linewidth=2)
axes[1, 1].set_title("Survival Rate by Family Size")
axes[1, 1].set_xlabel("Family Size")
axes[1, 1].set_ylabel("Survival Rate")
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Fare distribution
axes[1, 2].hist(df["Fare"], bins=40, color="steelblue", edgecolor="white")
axes[1, 2].set_title("Fare Distribution")
axes[1, 2].set_xlabel("Fare (£)")
axes[1, 2].set_ylabel("Passengers")

plt.tight_layout()
plt.show()
```

### Step 5 — NumPy Analysis

```python
# Use NumPy to compute statistics directly on arrays
survived = df[df["Survived"] == 1]["Age"].dropna().values
not_survived = df[df["Survived"] == 0]["Age"].dropna().values

print("=== Age Statistics ===")
print(f"Survivors — Mean: {np.mean(survived):.1f}, Std: {np.std(survived):.1f}")
print(f"Non-survivors — Mean: {np.mean(not_survived):.1f}, Std: {np.std(not_survived):.1f}")

# Correlation using NumPy
corr = np.corrcoef(df["Fare_norm"].values, df["Survived"].values)
print(f"\nCorrelation (Fare, Survived): {corr[0,1]:.3f}")
```

---

## Deliverable

Find:

1. What was the overall survival rate?
2. Which passenger class had the best survival odds? Why do you think that is?
3. How did gender affect survival?
4. Is there an ideal family size for survival?
5. What would you include as features if you were building a survival prediction model?

---

## Recommended Learning Resources

- **NumPy:** https://numpy.org/doc/stable/user/quickstart.html
- **Pandas:** https://pandas.pydata.org/docs/getting_started/10min.html
- **Matplotlib:** https://matplotlib.org/stable/tutorials/introductory/pyplot.html
- **Practice Datasets:** https://www.kaggle.com/datasets (Titanic, House Prices, Iris)
- **Free Course:** https://www.kaggle.com/learn (Pandas and Data Visualization courses are excellent)

---

*Guide version 1.0 · Python 3.9+ · NumPy 1.26+ · Pandas 2.0+ · Matplotlib 3.8+*
