import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np

path = r"images"
results_data = {
                "binary: straight-forward": rf"{path}\classification_results_binary straight forward classification.csv",
                "binary: detailed": rf"{path}\classification_results_binary detailed classification.csv",
                "binary: fill-in-the-blank": rf"{path}\classification_results_binary fill-in-blank.csv",
                "multiclass: straight-forward": rf"{path}\classification_results_multiclass straight forward classification.csv",
                "multiclass: detailed": rf"{path}\classification_results_multiclass detailed classification.csv",
                "multiclass: fill-in-the-blank":rf"{path}\classification_results_multiclass fill-in-blank.csv",
                }

group_class_map = {
                    "A":"scratches",
                    "B":"patches",
                    "C":"rolled-in_scale",
                    "D":"pitted_surface",
                    "E":"inclusion",
                    "F":"crazing"
                    }

# Load the data
data = {}
for key, file_path in results_data.items():
    data[key] = pd.read_csv(file_path)

# Preprocess the data
binary_data = {}
multiclass_data = {}
for key, df in data.items():
    if 'binary' in key:
        binary_data[key] = df
        binary_data[key]['Predicted Class'] = df['Predicted Group'].map(group_class_map)
    else:
        multiclass_data[key] = df
        multiclass_data[key]['Predicted Class'] = df['Predicted Group'].map(group_class_map)

# Calculate metrics
binary_metrics = {}
multiclass_metrics = {}
for key, df in binary_data.items():
    y_true = df['Actual Class']
    y_pred = df['Predicted Class']
    binary_metrics[key] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

for key, df in multiclass_data.items():
    y_true = df['Actual Class']
    y_pred = df['Predicted Class']
    multiclass_metrics[key] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=None, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=None, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=None, zero_division=0)
    }

print(binary_metrics)
print(multiclass_metrics)

# Visualization
# Bar chart for metrics
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Binary classification
ax = axs[0]
metrics = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(binary_metrics))
width = 0.2
for i, metric in enumerate(metrics):
    values = [binary_metrics[key][metric] for key in binary_metrics]
    ax.bar(x + i * width, values, width, label=metric)
ax.set_xticks(x + width / 2)
ax.set_xticklabels([key.split(':')[1].strip() for key in binary_metrics])
ax.set_xlabel('Prompting Strategy')
ax.set_ylabel('Score')
ax.set_ylim((0,1))
ax.set_title('Binary Classification Metrics')
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

# Multiclass classification
ax = axs[1]
x = np.arange(len(multiclass_metrics))
width = 0.2
for i, metric in enumerate(metrics):
    values = [multiclass_metrics[key][metric] for key in multiclass_metrics]
    ax.bar(x + i * width, values, width, label=metric)
ax.set_xticks(x + width / 2)
ax.set_xticklabels([key.split(':')[1].strip() for key in multiclass_metrics])
ax.set_xlabel('Prompting Strategy')
ax.set_ylabel('Score')
ax.set_title('Multiclass Classification Metrics')
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

plt.tight_layout()
plt.show()

# ROC curve
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

from sklearn.multiclass import OneVsRestClassifier
from sklearn.dummy import DummyClassifier

# Binary classification
ax = axs[0]
for key, df in binary_data.items():
    y_true = df['Actual Class'].map({'scratches': 1, 'patches': 0}).values
    y_score = df['Predicted Class'].map({'scratches': 1, 'patches': 0}).values  # Assuming inference time is a proxy for the score
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{key.split(':')[1].strip()} (AUC = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Binary Classification ROC Curve')
ax.legend()


# Bar chart for inference time and tokens
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Binary classification
ax = axs[0]
x = np.arange(len(binary_data))
width = 0.35
inference_time = [df['Inference Time'].mean() for df in binary_data.values()]
tokens = [df['Input Tokens'].mean() + df['Output Tokens'].mean() for df in binary_data.values()]

print(inference_time)
print(tokens)

# Scale the y-axis limits independently
inference_time_range = max(inference_time) - min(inference_time)
tokens_range = max(tokens) - min(tokens)
inference_time_min = min(inference_time) - inference_time_range * 0.1
inference_time_max = max(inference_time) + inference_time_range * 0.1
tokens_min = min(tokens) - tokens_range * 0.1
tokens_max = max(tokens) + tokens_range * 0.1

ax.bar(x - width/2, inference_time, width, label='Inference Time')
ax.set_xticks(x)
ax.set_xticklabels([key.split(':')[1].strip() for key in binary_data])
ax.set_xlabel('Prompting Strategy')
ax.set_ylabel('Average Inference Time (s)')
ax.set_ylim(inference_time_min, inference_time_max)
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

ax2 = ax.twinx()
ax2.bar(x + width/2, tokens, width, label='Tokens', color='orange')
ax2.set_ylabel('Average Tokens')
ax2.set_ylim(tokens_min, tokens_max)
ax2.legend(loc='upper right', bbox_to_anchor=(1, .90), fontsize='small')

ax.set_title('Binary Classification Average Inference Time and Tokens')

# Multiclass classification
ax = axs[1]
x = np.arange(len(multiclass_data))
width = 0.35
inference_time = [df['Inference Time'].mean() for df in multiclass_data.values()]
tokens = [df['Input Tokens'].mean() + df['Output Tokens'].mean() for df in multiclass_data.values()]

print(inference_time)
print(tokens)

# Scale the y-axis limits independently
inference_time_range = max(inference_time) - min(inference_time)
tokens_range = max(tokens) - min(tokens)
inference_time_min = min(inference_time) - inference_time_range * 0.1
inference_time_max = max(inference_time) + inference_time_range * 0.1
tokens_min = min(tokens) - tokens_range * 0.1
tokens_max = max(tokens) + tokens_range * 0.1

ax.bar(x - width/2, inference_time, width, label='Inference Time')
ax.set_xticks(x)
ax.set_xticklabels([key.split(':')[1].strip() for key in multiclass_data])
ax.set_xlabel('Prompting Strategy')
ax.set_ylabel('Average Inference Time (s)')
ax.set_ylim(inference_time_min, inference_time_max)
ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')

ax2 = ax.twinx()
ax2.bar(x + width/2, tokens, width, label='Tokens', color='orange')
ax2.set_ylabel('Average Tokens')
ax2.set_ylim(tokens_min, tokens_max)
ax2.legend(loc='upper right', bbox_to_anchor=(1, .90), fontsize='small')

ax.set_title('Multiclass Classification Average Inference Time and Tokens')

plt.tight_layout()
plt.show()