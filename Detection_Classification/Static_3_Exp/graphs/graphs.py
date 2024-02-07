
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

from Acoustic import utils





def overall_accuracy():
    experiment = Experiment()

    data = experiment.first_run_df

    # Overall Accuracy ----------------
    spec_70_9k_overall_detection_accuracy = int(np.round(np.mean(data[(data['feature'] == 'spectral') & (data['params'] == '70-9000')].get('overall'))))
    spec_350_3k_overall_detection_accuracy = int(np.round(np.mean(data[(data['feature'] == 'spectral') & (data['params'] == '350-3000')].get('overall'))))
    mfcc_13_overall_detection_accuracy = int(np.round(np.mean(data[(data['feature'] == 'mfcc') & (data['params'] == '13')].get('overall'))))
    mfcc_40_overall_detection_accuracy = int(np.round(np.mean(data[(data['feature'] == 'mfcc') & (data['params'] == '40')].get('overall'))))
    spec_70_9k_overall_detection_accuracy_neg = int(np.round(np.mean(data[(data['feature'] == 'spectral') & (data['params'] == '70-9000')].get('negative'))))
    spec_350_3k_overall_detection_accuracy_neg = int(np.round(np.mean(data[(data['feature'] == 'spectral') & (data['params'] == '350-3000')].get('negative'))))
    mfcc_13_overall_detection_accuracy_neg = int(np.round(np.mean(data[(data['feature'] == 'mfcc') & (data['params'] == '13')].get('negative'))))
    mfcc_40_overall_detection_accuracy_neg = int(np.round(np.mean(data[(data['feature'] == 'mfcc') & (data['params'] == '40')].get('negative'))))


    # Accuracy By Test ----------------
    spec_70_9k_overall_detection_accuracy_test1 = int(
        np.round(np.mean(
            data[(data['feature'] == 'spectral') & (data['params'] == '70-9000') & (data['test'] == 1)].get(
                'overall'))))
    spec_350_3k_overall_detection_accuracy_test1 = int(
        np.round(np.mean(
            data[(data['feature'] == 'spectral') & (data['params'] == '350-3000') & (data['test'] == 1)].get(
                'overall'))))
    mfcc_13_overall_detection_accuracy_test1 = int(
        np.round(np.mean(data[(data['feature'] == 'mfcc') & (data['params'] == '13') & (data['test'] == 1)].get('overall'))))
    mfcc_40_overall_detection_accuracy_test1 = int(
        np.round(np.mean(data[(data['feature'] == 'mfcc') & (data['params'] == '40') & (data['test'] == 1)].get('overall'))))

    spec_70_9k_overall_detection_accuracy_test2 = int(
        np.round(np.mean(
            data[(data['feature'] == 'spectral') & (data['params'] == '70-9000') & (data['test'] == 2)].get(
                'overall'))))
    spec_350_3k_overall_detection_accuracy_test2 = int(
        np.round(np.mean(
            data[(data['feature'] == 'spectral') & (data['params'] == '350-3000') & (data['test'] == 2)].get(
                'overall'))))
    mfcc_13_overall_detection_accuracy_test2 = int(
        np.round(np.mean(data[(data['feature'] == 'mfcc') & (data['params'] == '13') & (data['test'] == 2)].get('overall'))))
    mfcc_40_overall_detection_accuracy_test2 = int(
        np.round(np.mean(data[(data['feature'] == 'mfcc') & (data['params'] == '40') & (data['test'] == 2)].get('overall'))))

    # Negatives ----------------
    spec_70_9k_overall_detection_accuracy_test1_neg = int(
        np.round(np.mean(
            data[(data['feature'] == 'spectral') & (data['params'] == '70-9000') & (data['test'] == 1)].get(
                'negative'))))
    spec_350_3k_overall_detection_accuracy_test1_neg = int(
        np.round(np.mean(
            data[(data['feature'] == 'spectral') & (data['params'] == '350-3000') & (data['test'] == 1)].get(
                'negative'))))
    mfcc_13_overall_detection_accuracy_test1_neg = int(
        np.round(
            np.mean(data[(data['feature'] == 'mfcc') & (data['params'] == '13') & (data['test'] == 1)].get('negative'))))
    mfcc_40_overall_detection_accuracy_test1_neg = int(
        np.round(
            np.mean(data[(data['feature'] == 'mfcc') & (data['params'] == '40') & (data['test'] == 1)].get('negative'))))

    spec_70_9k_overall_detection_accuracy_test2_neg = int(
        np.round(np.mean(
            data[(data['feature'] == 'spectral') & (data['params'] == '70-9000') & (data['test'] == 2)].get(
                'negative'))))
    spec_350_3k_overall_detection_accuracy_test2_neg = int(
        np.round(np.mean(
            data[(data['feature'] == 'spectral') & (data['params'] == '350-3000') & (data['test'] == 2)].get(
                'negative'))))
    mfcc_13_overall_detection_accuracy_test2_neg = int(
        np.round(
            np.mean(data[(data['feature'] == 'mfcc') & (data['params'] == '13') & (data['test'] == 2)].get('negative'))))
    mfcc_40_overall_detection_accuracy_test2_neg = int(
        np.round(
            np.mean(data[(data['feature'] == 'mfcc') & (data['params'] == '40') & (data['test'] == 2)].get('negative'))))


    labels = ['Spec: 70-9k', 'Spec: 350-3k', 'MFCC: 13', 'MFCC: 40']
    overall = [spec_70_9k_overall_detection_accuracy, spec_350_3k_overall_detection_accuracy, mfcc_13_overall_detection_accuracy, mfcc_40_overall_detection_accuracy]
    test1_overall = [spec_70_9k_overall_detection_accuracy_test1, spec_350_3k_overall_detection_accuracy_test1, mfcc_13_overall_detection_accuracy_test1, mfcc_40_overall_detection_accuracy_test1]
    test2_overall = [spec_70_9k_overall_detection_accuracy_test2, spec_350_3k_overall_detection_accuracy_test2, mfcc_13_overall_detection_accuracy_test2, mfcc_40_overall_detection_accuracy_test2]
    overall_neg = [spec_70_9k_overall_detection_accuracy_neg, spec_350_3k_overall_detection_accuracy_neg, mfcc_13_overall_detection_accuracy_neg, mfcc_40_overall_detection_accuracy_neg]
    test1_neg = [spec_70_9k_overall_detection_accuracy_test1_neg, spec_350_3k_overall_detection_accuracy_test1_neg, mfcc_13_overall_detection_accuracy_test1_neg, mfcc_40_overall_detection_accuracy_test1_neg]
    test2_neg = [spec_70_9k_overall_detection_accuracy_test2_neg, spec_350_3k_overall_detection_accuracy_test2_neg, mfcc_13_overall_detection_accuracy_test2_neg, mfcc_40_overall_detection_accuracy_test2_neg]


    colors = ['#306699', 'green', 'dimgray', 'purple']

    rows = 1
    columns = 3
    gridlines = [20,30,40,50,60,70,80,90,100]

    fig, axes = plt.subplots(rows, columns, figsize=(13, 3))
    fig.suptitle('Feature Accuracy Overview: 10-40m', size=16)

    axes[0].set_title('Overall')
    axes[1].set_title('Test 1')
    axes[2].set_title('Test 2')

    axes[0].set_ylabel('Accuracy %')
    for i in range(columns):
        axes[i].set_ylim([20, 100])
        axes[i].tick_params(axis='x', rotation=0, labelsize=10)
        axes[i].set_yticks(gridlines)
        for line in gridlines:
            axes[i].axhline(line, c='gray', linestyle='-', zorder=0, linewidth=0.5, alpha=0.8)

    axes[0].bar(labels, overall, color=colors)
    axes[1].bar(labels, test1_overall, color=colors)
    axes[2].bar(labels, test2_overall, color=colors)

    # Function to add text on top of each bar
    def add_bar_labels(ax, data):
        for idx, value in enumerate(data):
            offset = value * 0.05
            position = value - offset
            ax.text(idx, position, f'{str(round(value, 2))}%',
                    ha='center', va='top', fontsize=10, color='white')

    # Function to add text on top of each bar
    def add_bar_labels_2(ax, data):
        for idx, value in enumerate(data):
            # Ensure the value is positive for the offset calculation
            if value > 0:
                offset = max(1, value * 0.22)  # Calculates a small offset from the bottom
                position = offset
            else:
                position = 0  # For negative or zero values, place label at the bottom

            ax.text(idx, position, f'Neg: {str(round(value, 2))}%',
                    ha='center', va='bottom', fontsize=8, color='white')

    # Add labels to bars in each subplot
    add_bar_labels(axes[0], overall)
    add_bar_labels(axes[1], test1_overall)
    add_bar_labels(axes[2], test2_overall)
    add_bar_labels_2(axes[0], overall_neg)
    add_bar_labels_2(axes[1], test1_neg)
    add_bar_labels_2(axes[2], test2_neg)

    plt.tight_layout(pad=1)
    plt.show()

def feature_comp():
    experiment = Experiment()
    data = experiment.first_run_df

    # Define features and their parameters for easy iteration
    features_params = [
        ('spectral', '70-9000'),
        ('spectral', '350-3000'),
        ('mfcc', '13'),
        ('mfcc', '40')
    ]
    distances = ['10m', '20m', '30m', '40m']
    n_groups = len(distances)
    tests = [1, 2]
    colors = ['#306699', 'green', 'dimgray', 'purple']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle('Feature Accuracy vs Distance', size=16)

    bar_width = 0.2
    index = np.arange(n_groups)

    for test in tests:
        test_index = tests.index(test)
        for feature, params in features_params:
            means = [
                int(np.mean(data[(data['feature'] == feature) & (data['params'] == params) & (data['test'] == test)].get(distance)))
                for distance in distances
            ]
            offset = (features_params.index((feature, params)) - len(features_params)/2) * bar_width + bar_width/2
            if feature == 'mfcc':
                feat_label = 'MFCC'
                parm_label = params
            else:
                feat_label = 'Spec'
                if params == '70-9000':
                    parm_label = '70-9k'
                else: parm_label = '350-9k'

            axes[test_index].bar(index + offset, means, bar_width, color=colors[features_params.index((feature, params))], label=f'{feat_label}: {parm_label}', edgecolor='white', linewidth=0.5)


        axes[test_index].set_title(f'Test {test}')
        axes[test_index].set_xlabel('Distance')
        axes[test_index].set_xticks(index + bar_width)
        axes[test_index].set_xticklabels(distances)
        axes[test_index].legend()
        gridlines = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        axes[test_index].set_yticks(gridlines)
        for line in gridlines:
            axes[test_index].axhline(line, c='gray', linestyle='-', zorder=0, linewidth=0.5, alpha=0.8)

    # Function to add value labels inside the bars
    def add_value_labels_inside(ax, spacing=5):
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            space = -spacing
            va = 'bottom'

            if y_value > 0:
                position = y_value + space
            else:
                position = space

            label = "{}%".format(int(y_value))
            ax.annotate(label, (x_value, position), xytext=(0, space), textcoords="offset points", ha='center', va=va, color='white', fontsize=7.5)

    axes[0].set_ylabel('Accuracy %')
    add_value_labels_inside(axes[0])
    add_value_labels_inside(axes[1])

    plt.tight_layout(pad=1)
    plt.show()

def accuracy_vs_samplength():
    samp_length = [2,3,4,5,6,7,8,9,10]

    experiment = Experiment()
    data = experiment.first_run_df

    # Accuracy By Test ----------------
    spec_70_9k_overall_detection_accuracy_test1 = data[(data['feature'] == 'spectral') & (data['params'] == '70-9000') & (data['test'] == 1)].get('overall')
    spec_350_3k_overall_detection_accuracy_test1 = data[(data['feature'] == 'spectral') & (data['params'] == '350-3000') & (data['test'] == 1)].get('overall')
    mfcc_13_overall_detection_accuracy_test1 = data[(data['feature'] == 'mfcc') & (data['params'] == '13') & (data['test'] == 1)].get('overall')
    mfcc_40_overall_detection_accuracy_test1 = data[(data['feature'] == 'mfcc') & (data['params'] == '40') & (data['test'] == 1)].get('overall')

    spec_70_9k_overall_detection_accuracy_test2 = data[(data['feature'] == 'spectral') & (data['params'] == '70-9000') & (data['test'] == 2)].get('overall')
    spec_350_3k_overall_detection_accuracy_test2 = data[(data['feature'] == 'spectral') & (data['params'] == '350-3000') & (data['test'] == 2)].get('overall')
    mfcc_13_overall_detection_accuracy_test2 = data[(data['feature'] == 'mfcc') & (data['params'] == '13') & (data['test'] == 2)].get('overall')
    mfcc_40_overall_detection_accuracy_test2 = data[(data['feature'] == 'mfcc') & (data['params'] == '40') & (data['test'] == 2)].get('overall')

    colors = ['#306699', 'green', 'dimgray', 'purple']

    rows = 1
    columns = 2
    gridlines = [30, 40, 50, 60, 70, 80, 90, 100]

    fig, axes = plt.subplots(rows, columns, figsize=(10, 3))
    fig.suptitle('Detection Accuracy vs Sample Length', size=16)

    axes[0].set_title('Test 1')
    axes[1].set_title('Test 2')

    axes[0].set_ylabel('Accuracy %')
    axes[0].set_ylabel('')
    axes[0].set_xlabel('Sample Length (s)')
    axes[1].set_xlabel('Sample Length (s)')
    for i in range(columns):
        axes[i].set_ylim([30, 100])
        axes[i].tick_params(axis='x', rotation=0) #, labelsize=8
        axes[i].set_yticks(gridlines)
        for line in gridlines:
            axes[i].axhline(line, c='gray', linestyle='-', zorder=0, linewidth=0.5, alpha=0.8)

    axes[0].plot(samp_length, spec_70_9k_overall_detection_accuracy_test1, color=colors[0], marker='o', markersize=4, label='Spec: 70-9k')
    axes[0].plot(samp_length, spec_350_3k_overall_detection_accuracy_test1, color=colors[1], marker='o', markersize=4, label='Spec: 350-3k')
    axes[0].plot(samp_length, mfcc_13_overall_detection_accuracy_test1, color=colors[2], marker='o', markersize=4, label='MFCC: 13')
    axes[0].plot(samp_length, mfcc_40_overall_detection_accuracy_test1, color=colors[3], marker='o', markersize=4, label='MFCC: 40')
    axes[1].plot(samp_length, spec_70_9k_overall_detection_accuracy_test2, color=colors[0], marker='o', markersize=4)
    axes[1].plot(samp_length, spec_350_3k_overall_detection_accuracy_test2, color=colors[1], marker='o', markersize=4)
    axes[1].plot(samp_length, mfcc_13_overall_detection_accuracy_test2, color=colors[2], marker='o', markersize=4)
    axes[1].plot(samp_length, mfcc_40_overall_detection_accuracy_test2, color=colors[3], marker='o', markersize=4)

    legend_lines = [Line2D([0], [0], color=color, lw=3) for color in colors]
    legend_labels = ['Spec: 70-9k', 'Spec: 350-3k', 'MFCC: 13', 'MFCC: 40']
    axes[1].legend(handles=legend_lines, labels=legend_labels, loc='lower left', ncol=1)

    plt.tight_layout(pad=1)
    plt.show()

def model_consistency_worst():
    experiment = Experiment()
    data = experiment.second_run_df

    overall = data[(data['feature'] == 'mfcc')].get('overall')
    negative = data[(data['feature'] == 'mfcc')].get('negative')
    m10 = data[(data['feature'] == 'mfcc')].get('10m')
    m20 = data[(data['feature'] == 'mfcc')].get('20m')
    m30 = data[(data['feature'] == 'mfcc')].get('30m')
    m40 = data[(data['feature'] == 'mfcc')].get('40m')

    original_score = 35
    overall_max = np.max(overall)
    overall_min = np.min(overall)
    # negative_average = np.mean(negative)
    # m10_average = np.max(m10)

    colors = ['blue', 'green', 'purple']

    rows = 2
    columns = 3
    gridlines = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    fig, axes = plt.subplots(rows, columns, figsize=(12, 5))
    fig.suptitle('Model Consistency Histogram: MFCC at 7s on Test 1 ', size=16)

    axes[0][0].set_title('Overall Accuracy', size=12)
    axes[0][1].set_title('Negative Samples Accuracy', size=12)
    axes[0][2].set_title('10m Accuracy', size=12)
    axes[1][0].set_title('20m Accuracy', size=12)
    axes[1][1].set_title('30m Accuracy', size=12)
    axes[1][2].set_title('40m Accuracy', size=12)

    axes[0][0].hist(overall, bins=50, range=(0,100), color='#3977b4', edgecolor='#306699')
    axes[0][1].hist(negative, bins=50, range=(0,100), color='#3977b4', edgecolor='#306699')
    axes[0][2].hist(m10, bins=50, range=(0,100), color='#3977b4', edgecolor='#306699')
    axes[1][0].hist(m20, bins=50, range=(0,100), color='#3977b4', edgecolor='#306699')
    axes[1][1].hist(m30, bins=50, range=(0,100), color='#3977b4', edgecolor='#306699')
    axes[1][2].hist(m40, bins=50, range=(0,100), color='#3977b4', edgecolor='#306699')

    axes[0][0].axvline(original_score, c='red', linestyle='dotted', zorder=1, linewidth=2)

    for row in range(rows):
        for col in range(columns):
            axes[row][col].set_xlabel('Accuracy %')
            axes[row][col].set_xticks(gridlines)
            axes[row][col].set_yticks([])
            for line in gridlines:
                axes[row][col].axvline(line, c='gray', linestyle='-', zorder=0, linewidth=0.5, alpha=0.8)

    line = Line2D([0], [0], color='red', linestyle='dotted', linewidth=2, label=f'Original Score: {original_score}%')
    text_legend_1 = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0,
                                     label=f'Score Range: {overall_min}-{overall_max}%')
    text_legend_2 = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0,
                                     label=f'High Score: {overall_min}%')
    axes[0][0].legend(handles=[line, text_legend_1], loc='upper right', fontsize=8)

    plt.tight_layout(pad=1)
    plt.show()

def model_consistency_best():
    experiment = Experiment()
    data = experiment.second_run_df

    overall = data[(data['feature'] == 'spectral') & (data['params'] == '70-9000')].get('overall')
    negative = data[(data['feature'] == 'spectral') & (data['params'] == '70-9000')].get('negative')
    m10 = data[(data['feature'] == 'spectral') & (data['params'] == '70-9000')].get('10m')
    m20 = data[(data['feature'] == 'spectral') & (data['params'] == '70-9000')].get('20m')
    m30 = data[(data['feature'] == 'spectral') & (data['params'] == '70-9000')].get('30m')
    m40 = data[(data['feature'] == 'spectral') & (data['params'] == '70-9000')].get('40m')

    original_score = 98
    overall_max = np.max(overall)
    overall_min = np.min(overall)
    # negative_average = np.mean(negative)
    # m10_average = np.max(m10)

    colors = ['blue', 'green', 'purple']

    rows = 2
    columns = 3
    gridlines = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    fig, axes = plt.subplots(rows, columns, figsize=(12, 5))
    fig.suptitle('Model Consistency Histogram: Spectral 70-9kHz at 6s on Test 2 ', size=16)

    axes[0][0].set_title('Overall Accuracy', size=12)
    axes[0][1].set_title('Negative Samples Accuracy', size=12)
    axes[0][2].set_title('10m Accuracy', size=12)
    axes[1][0].set_title('20m Accuracy', size=12)
    axes[1][1].set_title('30m Accuracy', size=12)
    axes[1][2].set_title('40m Accuracy', size=12)

    axes[0][0].hist(overall, bins=50, range=(0, 100), color='#3977b4', edgecolor='#306699')
    axes[0][1].hist(negative, bins=50, range=(0, 100), color='#3977b4', edgecolor='#306699')
    axes[0][2].hist(m10, bins=50, range=(0, 100), color='#3977b4', edgecolor='#306699')
    axes[1][0].hist(m20, bins=50, range=(0, 100), color='#3977b4', edgecolor='#306699')
    axes[1][1].hist(m30, bins=50, range=(0, 100), color='#3977b4', edgecolor='#306699')
    axes[1][2].hist(m40, bins=50, range=(0, 100), color='#3977b4', edgecolor='#306699')

    axes[0][0].axvline(original_score, c='red', linestyle='dotted', zorder=1, linewidth=2)

    for row in range(rows):
        for col in range(columns):
            axes[row][col].set_xlabel('Accuracy %')
            axes[row][col].set_xticks(gridlines)
            axes[row][col].set_yticks([])
            for line in gridlines:
                axes[row][col].axvline(line, c='gray', linestyle='-', zorder=0, linewidth=0.5, alpha=0.8)

    line = Line2D([0], [0], color='red', linestyle='dotted', linewidth=2, label=f'Original Score: {original_score}%')
    text_legend_1 = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0,
                                       label=f'Score Range: {overall_min}-{overall_max}%')
    text_legend_2 = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0,
                                       label=f'High Score: {overall_min}%')
    axes[0][0].legend(handles=[line, text_legend_1], loc='upper left', fontsize=8)

    plt.tight_layout(pad=1)
    plt.show()

def accuracy_vs_nperseg():

    experiment = Experiment()
    data = experiment.nperseg_df

    # Accuracy By Test ----------------
    # mfcc_overall_detection_accuracy_test2 = data[(data['feature'] == 'mfcc') & (data['test'] == 2)].get('overall')
    nps_512_overall = data[data['nperseg'] == 512].get('overall')
    nps_1024_overall = data[data['nperseg'] == 1024].get('overall')
    nps_2048_overall = data[data['nperseg'] == 2048].get('overall')
    nps_4096_overall = data[data['nperseg'] == 4096].get('overall')
    nps_8192_overall = data[data['nperseg'] == 8192].get('overall')
    nps_16384_overall = data[data['nperseg'] == 16384].get('overall')


    nps_512_mean = int(np.mean(nps_512_overall))
    nps_1024_mean = int(np.mean(nps_1024_overall))
    nps_2048_mean = int(np.mean(nps_2048_overall))
    nps_4096_mean = int(np.mean(nps_4096_overall))
    nps_8192_mean = int(np.mean(nps_8192_overall))
    nps_16384_mean = int(np.mean(nps_16384_overall))

    rows = 1
    columns = 1
    gridlines = [50, 60, 70, 80, 90, 100]

    fig, axes = plt.subplots(rows, columns, figsize=(10, 3))
    fig.suptitle('Spectrogram Frequency Resolution vs Accuracy', size=12)

    # axes.set_title('Freq Resolution')
    axes.set_ylabel('Accuracy %')
    axes.set_xlabel('nperseg')

    axes.set_ylim([50, 100])
    axes.tick_params(axis='x', rotation=0)  # , labelsize=8
    axes.set_yticks(gridlines)
    for line in gridlines:
        axes.axhline(line, c='gray', linestyle='-', zorder=0, linewidth=0.5, alpha=0.8)

    freq_colors = ['slateblue', 'mediumseagreen', 'plum', 'darkgrey', 'powderblue', 'lightcoral']
    freq_colors = ['#306699', '#306699', '#306699', '#306699', '#306699', '#306699']
    # freq_colors = ['slateblue', 'mediumseagreen', 'lightcoral', 'darkgrey', 'powderblue', 'plum']

    freq_acc = [nps_512_mean, nps_1024_mean, nps_2048_mean, nps_4096_mean, nps_8192_mean, nps_16384_mean]
    freq_res = ['512', '1024', '2048', '4096', '8192', '16384']

    axes.bar(freq_res, freq_acc, color=freq_colors)

    # Function to add value labels inside the bars
    def add_value_labels_inside(ax, spacing=5):
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            space = -spacing
            va = 'bottom'

            if y_value > 0:
                position = y_value + space
            else:
                position = space

            label = "{}%".format(int(y_value))
            ax.annotate(label, (x_value, position), xytext=(0, space), textcoords="offset points", ha='center', va=va,
                        color='white', fontsize=9)

    add_value_labels_inside(axes)
    add_value_labels_inside(axes)

    plt.tight_layout(pad=1)
    plt.show()

def feature_size():
    experiment = Experiment()
    data = experiment.accuracy_data_df

    sizes = data.get('size')
    overall = data.get('overall')
    features = data.get('feature')
    feature_colors = np.where(features == 'mfcc', 'purple', 'g')  # #306699'r' for 'mfcc', 'b' for 'spectral'

    rows = 1
    columns = 1
    gridlines = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    fig, axes = plt.subplots(rows, columns, figsize=(10, 4))
    fig.suptitle('Feature Size vs Accuracy', size=16)

    axes.set_ylabel('Accuracy %')
    axes.set_xlabel('Size')

    axes.set_ylim([20, 100])
    axes.tick_params(axis='x', rotation=0)  # , labelsize=8
    axes.set_yticks(gridlines)
    for line in gridlines:
        axes.axhline(line, c='gray', linestyle='-', zorder=0, linewidth=0.5, alpha=0.8)

    axes.scatter(sizes, overall, color=feature_colors, s=8)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='MFCC',
                              markerfacecolor='purple', markersize=8),
                       Line2D([0], [0], marker='o', color='w', label='Spectral',
                              markerfacecolor='g', markersize=8)]
    axes.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout(pad=1)
    plt.show()

def feature_size_3d():
    experiment = Experiment()
    data = experiment.accuracy_data_df

    sizes = data.get('size')
    lengths = data.get('length')
    overall = data.get('overall')
    features = data.get('feature')

    feature_colors = np.where(features == 'mfcc', 'r', 'b')  # 'r' for 'mfcc', 'b' for 'spectral'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the scatter plot
    ax.scatter(sizes, lengths, overall, c=feature_colors)


    # Setting labels
    ax.set_xlabel('Sizes')
    ax.set_ylabel('Lengths')
    ax.set_zlabel('Accuracy')

    ax.set_zlim(0, 100)

    # from matplotlib.lines import Line2D  # Import Line2D for custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='mfcc',
                              markerfacecolor='r', markersize=8),
                       Line2D([0], [0], marker='o', color='w', label='spectral',
                              markerfacecolor='b', markersize=8)]
    ax.legend(handles=legend_elements, loc='best')

    plt.show()


    # stats = kwargs.get('stats', False)
    # if stats:
    #     print(f'Neg Overall Mean: {negative_mean}')
    #     print(f'512 Overall Mean: {nps_512_mean}')
    #     print(f'1024 Overall Mean: {nps_1024_mean}')
    #     print(f'2048 Overall Mean: {nps_2048_mean}')
    #     print(f'4096 Overall Mean: {nps_4096_mean}')
    #
    #
    # colors = ['blue', 'green', 'purple']
    #
    # rows = 1
    # columns = 1
    # gridlines = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #
    # fig, axes = plt.subplots(rows, columns, figsize=(10, 5))
    # fig.suptitle('Spectrogram: Num per Seg', size=12)
    #
    # axes.set_title('Freq Resolution')
    # axes.set_ylabel('Accuracy %')
    # axes.set_xlabel('nperseg')
    #
    # axes.set_ylim([0, 100])
    # axes.tick_params(axis='x', rotation=0)  # , labelsize=8
    # axes.set_yticks(gridlines)
    # for line in gridlines:
    #     axes.axhline(line, c='gray', linestyle='-', zorder=0, linewidth=0.5, alpha=0.8)
    #
    # freq_colors = ['navy', 'green', 'indigo', 'teal']
    # freq_acc = [nps_512_mean, nps_1024_mean, nps_2048_mean, nps_4096_mean]
    # freq_res = ['512', '1024', '2048', '4096']
    #
    # axes.bar(freq_res, freq_acc, color=freq_colors)
    #
    # plt.tight_layout(pad=1)
    #
    # # Save or display the plot
    # save = kwargs.get('save', False)
    # save_path = kwargs.get('save_path', '')
    # if save:
    #     plt.savefig(f'{save_path}/overall_accuracy.png')
    #     plt.close(fig)
    # else:
    #     plt.show()

def target_sample_bias():
    experiment = Experiment()
    data = experiment.accuracy_data_df

    # Define features and their parameters for easy iteration
    features_params = [
        ('spectral', '70-9000'),
        ('spectral', '350-3000'),
        ('mfcc', '13'),
        ('mfcc', '40')
    ]

    distances = ['10m', '20m', '30m', '40m']
    n_groups = len(distances)
    tests = [1, 2]
    colors = ['#306699', 'green', 'dimgray', 'purple']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle('Feature Accuracy vs Distance', size=16)

    bar_width = 0.2
    index = np.arange(n_groups)

    for test in tests:
        test_index = tests.index(test)
        for feature, params in features_params:
            means = [
                int(np.mean(data[(data['feature'] == feature) & (data['params'] == params) & (data['test'] == test)].get(distance)))
                for distance in distances
            ]
            offset = (features_params.index((feature, params)) - len(features_params)/2) * bar_width + bar_width/2
            if feature == 'mfcc':
                feat_label = 'MFCC'
                parm_label = params
            else:
                feat_label = 'Spec'
                if params == '70-9000':
                    parm_label = '70-9k'
                else: parm_label = '350-9k'

            axes[test_index].bar(index + offset, means, bar_width, color=colors[features_params.index((feature, params))], label=f'{feat_label}: {parm_label}', edgecolor='white', linewidth=0.5)


        axes[test_index].set_title(f'Test {test}')
        axes[test_index].set_xlabel('Distance')
        axes[test_index].set_xticks(index + bar_width)
        axes[test_index].set_xticklabels(distances)
        axes[test_index].legend()
        gridlines = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        axes[test_index].set_yticks(gridlines)
        for line in gridlines:
            axes[test_index].axhline(line, c='gray', linestyle='-', zorder=0, linewidth=0.5, alpha=0.8)

    # Function to add value labels inside the bars
    def add_value_labels_inside(ax, spacing=5):
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            space = -spacing
            va = 'bottom'

            if y_value > 0:
                position = y_value + space
            else:
                position = space

            label = "{}%".format(int(y_value))
            ax.annotate(label, (x_value, position), xytext=(0, space), textcoords="offset points", ha='center', va=va, color='white', fontsize=7.5)

    axes[0].set_ylabel('Accuracy %')
    add_value_labels_inside(axes[0])
    add_value_labels_inside(axes[1])

    plt.tight_layout(pad=1)
    plt.show()




class Experiment:
    def __init__(self):
        base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
        accuracy_SL_filepath = f'{base_path}/Detection_Classification/Static_3_Exp/graphs/csv/accuracy_samplen.csv'
        model_confidence_filepath = f'{base_path}/Detection_Classification/Static_3_Exp/graphs/csv/model_confidence.csv'
        accuarcy_nperseg_filepath = f'{base_path}/Detection_Classification/Static_3_Exp/graphs/csv/accuracy_nperseg.csv'

        self.first_run_csv = utils.CSVFile(accuracy_SL_filepath)
        self.first_run_df = pd.read_csv(accuracy_SL_filepath)
        self.second_run_csv = utils.CSVFile(model_confidence_filepath)
        self.second_run_df = pd.read_csv(model_confidence_filepath)

        self.nperseg_df = pd.read_csv(accuarcy_nperseg_filepath)

        base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
        accuracy_SL_filepath = f'{base_path}/Detection_Classification/Static_3_Exp/graphs/csv/Model_Accuracy.csv'
        self.accuracy_data_df = pd.read_csv(accuracy_SL_filepath)




if __name__ == '__main__':
    # overall_accuracy()
    # feature_comp()
    # accuracy_vs_samplength()
    # accuracy_vs_nperseg()
    # model_consistency_worst()
    # model_consistency_best()
    # feature_size()
    # feature_size_3d()
    # mfcc_comp()
    # spec_comp()
    target_sample_bias()
