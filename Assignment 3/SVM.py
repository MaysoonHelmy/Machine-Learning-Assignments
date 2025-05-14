import os
import glob
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Dataset
train =r'C:\Users\dell\Desktop\ML\Machine Learning\Assignment 3\Assignment dataset\train'  
test = r'C:\Users\dell\Desktop\ML\Machine Learning\Assignment 3\Assignment dataset\test'   

# Classes
classes = ['accordian', 'dollar_bill', 'motorbike', 'soccer_ball']
class_to_label = {cls: idx for idx, cls in enumerate(classes)}
label_to_class = {idx: cls for cls, idx in class_to_label.items()}

X_train = []
y_train = []
X_test = []
y_test = []

# Load and preprocess training data
for cls in classes:
    cls_dir = os.path.join(train, cls)
    for img_file in glob.glob(os.path.join(cls_dir, '*.jpg')):
        
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(image, (128, 64))
        fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
        
        X_train.append(fd)
        y_train.append(class_to_label[cls])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Load and preprocess test data
for cls in classes:
    cls_dir = os.path.join(test, cls)
    for img_file in glob.glob(os.path.join(cls_dir, '*.jpg')):
        
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(image, (128, 64))
        fd, hog_image_ = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
        
        X_test.append(fd)
        y_test.append(class_to_label[cls])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Parameter Grids
param_grid_linear = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'class_weight': ['balanced', None]
}
param_grid_rbf = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'class_weight': ['balanced', None]
}
param_grid_poly = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'degree': [1,2, 3, 4],
    'coef0': [0, 1],
    'class_weight': ['balanced', None]
}

# SVM with linear kernel
print("\nSVM with linear kernel:")
print("-" * 50)
linear_svm = SVC(kernel='linear', probability=True)
linear_grid_search = GridSearchCV(linear_svm, param_grid_linear, cv=5, scoring='accuracy', n_jobs=-1)
linear_grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters for linear kernel: {linear_grid_search.best_params_}")

linear_best_model = linear_grid_search.best_estimator_
linear_train_accuracy = linear_best_model.score(X_train_scaled, y_train)
y_pred_linear = linear_best_model.predict(X_test_scaled)
linear_test_accuracy = accuracy_score(y_test, y_pred_linear)

print(f"Linear Kernel - Training Accuracy: {linear_train_accuracy * 100:.2f}%")
print(f"Linear Kernel - Testing Accuracy: {linear_test_accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred_linear, target_names=classes, zero_division=0))

# SVM with RBF kernel
print("\nSVM with RBF kernel:")
print("-" * 50)
rbf_svm = SVC(kernel='rbf', probability=True)
rbf_grid_search = GridSearchCV(rbf_svm, param_grid_rbf, cv=5, scoring='accuracy', n_jobs=-1)
rbf_grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters for RBF kernel: {rbf_grid_search.best_params_}")

rbf_best_model = rbf_grid_search.best_estimator_
rbf_train_accuracy = rbf_best_model.score(X_train_scaled, y_train)
y_pred_rbf = rbf_best_model.predict(X_test_scaled)
rbf_test_accuracy = accuracy_score(y_test, y_pred_rbf)

print(f"RBF Kernel - Training Accuracy: {rbf_train_accuracy * 100:.2f}%")
print(f"RBF Kernel - Testing Accuracy: {rbf_test_accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred_rbf, target_names=classes, zero_division=0))

# SVM with polynomial kernel
print("\nSVM with polynomial kernel:")
print("-" * 50)
poly_svm = SVC(kernel='poly', probability=True)
poly_grid_search = GridSearchCV(poly_svm, param_grid_poly, cv=5, scoring='accuracy', n_jobs=-1)
poly_grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters for polynomial kernel: {poly_grid_search.best_params_}")

poly_best_model = poly_grid_search.best_estimator_
poly_train_accuracy = poly_best_model.score(X_train_scaled, y_train)
y_pred_poly = poly_best_model.predict(X_test_scaled)
poly_test_accuracy = accuracy_score(y_test, y_pred_poly)

print(f"Polynomial Kernel - Training Accuracy: {poly_train_accuracy * 100:.2f}%")
print(f"Polynomial Kernel - Testing Accuracy: {poly_test_accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred_poly, target_names=classes, zero_division=0))


