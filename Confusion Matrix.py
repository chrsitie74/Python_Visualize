import os
import numpy as np

import cv2
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CLASS_NAME = ['0', '1', '2']

def Confusion_Matrix():

    ''' Confusion Matrix '''
    confusion_matrix = np.zeros((len(_CLASS_NAME), len(_CLASS_NAME)))  # 0-axis: Prediction, 1-axis: Ground Truth
    confusion_matrix[0, 0] = 100
    confusion_matrix[0, 1] = 24
    confusion_matrix[0, 2] = 5
    confusion_matrix[1, 0] = 3
    confusion_matrix[1, 1] = 153
    confusion_matrix[1, 2] = 15
    confusion_matrix[2, 0] = 25
    confusion_matrix[2, 1] = 83
    confusion_matrix[2, 2] = 203

    normalized_confusion_matrix = confusion_matrix.copy()
    normalized_confusion_matrix[:, 0] = normalized_confusion_matrix[:, 0] / sum(normalized_confusion_matrix[:, 0])
    normalized_confusion_matrix[:, 1] = normalized_confusion_matrix[:, 1] / sum(normalized_confusion_matrix[:, 1])
    normalized_confusion_matrix[:, 2] = normalized_confusion_matrix[:, 2] / sum(normalized_confusion_matrix[:, 2])
  
    print(confusion_matrix)
    print(normalized_confusion_matrix)
    normalized_confusion_matrix = normalized_confusion_matrix * 100
  normalized_confusion_matrix = normalized_confusion_matrix * 100
normalized_confusion_matrix = normalized_confusion_matrix * 100
normalized_confusion_matrix = normalized_confusion_matrix * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(confusion_matrix.astype(np.int64), annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=_CLASS_NAME, yticklabels=_CLASS_NAME,
                annot_kws={'size': 12, 'weight': 'bold'})
    axes[0].set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Predicted', fontweight='bold')
    axes[0].set_xlabel('Ground Truth', fontweight='bold')

    cm1_labels = np.array([["{:.2f}%".format(value) for value in row] for row in normalized_confusion_matrix])
    sns.heatmap(normalized_confusion_matrix, annot=cm1_labels, fmt='', cmap='Blues', ax=axes[1],
                xticklabels=_CLASS_NAME, yticklabels=_CLASS_NAME,
                annot_kws={'size': 12, 'weight': 'bold'})
    axes[1].set_title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('Predicted', fontweight='bold')
    axes[1].set_xlabel('Ground Truth', fontweight='bold')
    
    font_prop = font_manager.FontProperties(weight='bold', size=10)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), fontproperties=font_prop)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontproperties=font_prop)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), fontproperties=font_prop)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), fontproperties=font_prop)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

if __name__=='__main__':
    Confusion_Matrix()
