```markdown
# Machine Learning Assignments

This repository contains three machine learning assignments. Each focuses on a different core concept using Python and popular ML libraries.

---

## 📌 Contents

### 1. Manual Gradient Descent
- **File**: `Manual Gradient Descent.py`
- **Goal**: Implement linear regression using gradient descent from scratch
- **Highlights**:
  - No ML libraries used for training
  - Feature selection and normalization included
  - Manual gradient calculation and loss tracking
  - Hyperparameter tuning for learning rate and epochs
- **Usage**: Best for understanding how linear models learn

---

### 2. Polynomial Regression
- **File**: `Polynomial Regression.py`
- **Goal**: Fit and evaluate polynomial regression models
- **Highlights**:
  - Custom feature selector and scaler
  - Recursive polynomial feature generator
  - Degree vs MSE visualization
  - Normalized MSE for fair comparison
- **Usage**: Learn how model complexity affects performance

---

### 3. SVM Image Classification
- **File**: `SVM.py`
- **Goal**: Classify images using Support Vector Machines
- **Highlights**:
  - Extracts HOG features from grayscale images
  - Uses `GridSearchCV` for hyperparameter tuning
  - Tests linear, RBF, and polynomial kernels
  - Reports training and test accuracy, classification report
- **Classes**:
  - `accordian`, `dollar_bill`, `motorbike`, `soccer_ball`
- **Dataset structure**:
```


## 🔧 Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn opencv-python scikit-image
````

---

## ▶️ Running

Each script is standalone:

```bash
python "Manual Gradient Descent.py"
python "Polynomial Regression.py"
python "SVM.py"
```

---

## 🧾 License

Academic use only

```
```
