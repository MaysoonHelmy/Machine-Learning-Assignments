import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.metrics import mean_squared_error

# load el data lel dataframe
data = pd.read_csv('assignment1dataset.csv')
features = ['NCustomersPerDay','AverageOrderValue','WorkingHoursPerDay','NEmployees','MarketingSpendPerDay','LocationFootTraffic']
target = 'RevenuePerDay'

results = []
best_feature = None
best_mse = float('inf')
best_params = {}

# Manual gradient descent like lab2 hands on
def gradient_descent(X, Y, lr, epochs):
    m = 0.0
    c = 0.0
    n = float(len(X))
    mse_values = []
    for _ in range(epochs):
        y_pred = m * X + c
        dm = (-2 / n) * np.sum((Y - y_pred) * X)
        dc = (-2 / n) * np.sum(Y - y_pred)
        m -= lr * dm
        c -= lr * dc
        mse_values.append(mean_squared_error(Y, y_pred))
    return m, c, mse_values

# lists of all the hyperparameters to trys
LR = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
Epochs = [0, 25, 50, 75, 100, 125, 150, 175, 200]

for feature_name in features:
    x_raw = data[feature_name].values
    x = (x_raw - np.mean(x_raw)) / np.std(x_raw)
    y = data[target].values

    feature_results = {'name': feature_name,'lr_results': {},'epoch_results': {},'optimal_lr': None,'optimal_epoch': None}

    for lr in LR :
        m, c, mse_vals = gradient_descent(x, y, lr, 100)
        if mse_vals:
            final_mse = mse_vals[-1]
            feature_results['lr_results'][lr] = final_mse

    if feature_results['lr_results']:
        optimal_lr = min(feature_results['lr_results'], key=feature_results['lr_results'].get)
        feature_results['optimal_lr'] = optimal_lr

        for e in Epochs:
            if e == 0:
                initial_pred = np.zeros_like(y)
                mse0 = mean_squared_error(y, initial_pred)
                feature_results['epoch_results'][e] = mse0
            else:
                m, c, mse_vals = gradient_descent(x, y, optimal_lr, e)
                if mse_vals:
                    final_mse = mse_vals[-1]
                    feature_results['epoch_results'][e] = final_mse 
                    
        if feature_results['epoch_results']:
            optimal_epoch = min(feature_results['epoch_results'], key=feature_results['epoch_results'].get)
            feature_results['optimal_epoch'] = optimal_epoch

            if feature_results['epoch_results'][optimal_epoch] < best_mse:
                best_mse = feature_results['epoch_results'][optimal_epoch]
                best_feature = feature_name
                best_params = {'lr': optimal_lr,'epoch': optimal_epoch,'mse': best_mse}

    results.append(feature_results)


all_mse_values = []
for res in results:
    all_mse_values.extend(list(res['lr_results'].values()))
    all_mse_values.extend(list(res['epoch_results'].values()))
max_mse_global = max(all_mse_values) if all_mse_values else 1


for res in results:
    print("\n" + res['name'])
    print("Learning Rate vs MSE")
    for lr in LR:
        mse_val = res['lr_results'].get(lr)
        if mse_val is not None:
            norm_val = mse_val / max_mse_global
            print("LR", lr, ":", f"{norm_val:.4f}")
    print("Epochs vs MSE")
    for e in Epochs:
        mse_val = res['epoch_results'].get(e)
        if mse_val is not None:
            norm_val = mse_val / max_mse_global
            print("Epochs", e, ":", f"{norm_val:.4f}")

print("\n=== Best Overall Feature ===")
print("Feature", best_feature)
print("Learning Rate", best_params.get('lr'))
print("Epochs", best_params.get('epoch'))
print("MSE", f"{best_params.get('mse') / max_mse_global:.4f}")

fig, axs = plt.subplots(2, 1, figsize=(18, 16), tight_layout=True)

colors = plt.cm.tab10(np.linspace(0, 1, len(features)))
line_styles = ['-', '-', '-', '-', '-', '-']

for i, res in enumerate(results):
    lr_items = sorted(res['lr_results'].items(), key=lambda x: x[0])
    if lr_items:
        x_vals = [item[0] for item in lr_items]
        y_vals = [item[1] / max_mse_global for item in lr_items]
        axs[0].plot(
            x_vals, y_vals,
            linestyle=line_styles[i % len(line_styles)],
            marker='o',
            markersize=6,
            color=colors[i],
            linewidth=2,
            label=res['name']
        )

axs[0].set_title("LR vs MSE", fontsize=14)
axs[0].set_xlabel("LR", fontsize=12)
axs[0].set_ylabel("MSE", fontsize=12)
axs[0].set_xticks(LR)
axs[0].legend(loc='upper right')
axs[0].grid(True)

for i, res in enumerate(results):
    epoch_items = sorted(res['epoch_results'].items(), key=lambda x: x[0])
    if epoch_items:
        x_vals = [item[0] for item in epoch_items]
        y_vals = [item[1] / max_mse_global for item in epoch_items]
        axs[1].plot(
            x_vals, y_vals,
            linestyle=line_styles[i % len(line_styles)],
            marker='o',
            markersize=6,
            color=colors[i],
            linewidth=2,
            label=res['name']
        )

axs[1].set_title("MSE vs Epochs", fontsize=14)
axs[1].set_xlabel("Epochs", fontsize=12)
axs[1].set_ylabel("MSE", fontsize=12)
axs[1].legend(loc='upper right')
axs[1].grid(True)

fig.suptitle("Hyperparameter Tuning", fontsize=16, y=0.99)

root = tk.Tk()
root.wm_title("Simple Linear Regression Assignment")
root.geometry("1200x900")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

canvas = FigureCanvasTkAgg(fig, master=main_frame)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

toolbar_frame = tk.Frame(main_frame)
toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
toolbar.update()

root.mainloop()
