
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

from Acoustic import utils



def overall_accuracy(**kwargs):
    experiment = Experiment()

    data = experiment.first_run_df

    # Overall Accuracy ----------------
    spec_70_9k_overall_detection_accuracy = int(np.round(np.mean(data[(data['feature'] == 'spectral') & (data['params'] == '70-9000')].get('overall'))))
    spec_350_3k_overall_detection_accuracy = int(np.round(np.mean(data[(data['feature'] == 'spectral') & (data['params'] == '350-3000')].get('overall'))))
    mfcc_overall_detection_accuracy = int(np.round(np.mean(data[data['feature'] == 'mfcc'].get('overall'))))

    # Accuracy By Test ----------------
    spec_70_9k_overall_detection_accuracy_test1 = int(
        np.round(np.mean(
            data[(data['feature'] == 'spectral') & (data['params'] == '70-9000') & (data['test'] == 1)].get(
                'overall'))))
    spec_350_3k_overall_detection_accuracy_test1 = int(
        np.round(np.mean(
            data[(data['feature'] == 'spectral') & (data['params'] == '350-3000') & (data['test'] == 1)].get(
                'overall'))))
    mfcc_overall_detection_accuracy_test1 = int(
        np.round(np.mean(data[(data['feature'] == 'mfcc') & (data['test'] == 1)].get('overall'))))

    spec_70_9k_overall_detection_accuracy_test2 = int(
        np.round(np.mean(
            data[(data['feature'] == 'spectral') & (data['params'] == '70-9000') & (data['test'] == 2)].get(
                'overall'))))
    spec_350_3k_overall_detection_accuracy_test2 = int(
        np.round(np.mean(
            data[(data['feature'] == 'spectral') & (data['params'] == '350-3000') & (data['test'] == 2)].get(
                'overall'))))
    mfcc_overall_detection_accuracy_test2 = int(
        np.round(np.mean(data[(data['feature'] == 'mfcc') & (data['test'] == 2)].get('overall'))))



    stats = kwargs.get('stats', False)
    if stats:
        print(f'Spec: 70-9k Overall Accuracy - {spec_70_9k_overall_detection_accuracy}')
        print(f'Spec: 350-3k Overall Accuracy - {spec_350_3k_overall_detection_accuracy}')
        print(f'MFCC: 13 Overall Accuracy - {mfcc_overall_detection_accuracy}')
        print('-'*40)
        print(f'Spec: 70-9k Overall Accuracy: Test 1 - {spec_70_9k_overall_detection_accuracy_test1}')
        print(f'Spec: 350-3k Overall Accuracy: Test 1 - {spec_350_3k_overall_detection_accuracy_test1}')
        print(f'MFCC: 13 Overall Accuracy: Test 1 - {mfcc_overall_detection_accuracy_test1}')
        print('-' * 40)
        print(f'Spec: 70-9k Overall Accuracy: Test 2 - {spec_70_9k_overall_detection_accuracy_test2}')
        print(f'Spec: 350-3k Overall Accuracy: Test 2 - {spec_350_3k_overall_detection_accuracy_test2}')
        print(f'MFCC: 13 Overall Accuracy: Test 2 - {mfcc_overall_detection_accuracy_test2}')
        print('-' * 40)

    labels = ['Spec: 70-9k', 'Spec: 350-3k', 'MFCC: 13']
    overall = [spec_70_9k_overall_detection_accuracy, spec_350_3k_overall_detection_accuracy, mfcc_overall_detection_accuracy]
    test1_overall = [spec_70_9k_overall_detection_accuracy_test1, spec_350_3k_overall_detection_accuracy_test1, mfcc_overall_detection_accuracy_test1]
    test2_overall = [spec_70_9k_overall_detection_accuracy_test2, spec_350_3k_overall_detection_accuracy_test2, mfcc_overall_detection_accuracy_test2]

    colors = ['blue', 'green', 'purple']

    rows = 1
    columns = 3
    gridlines = [0,10,20,30,40,50,60,70,80,90,100]

    fig, axes = plt.subplots(rows, columns, figsize=(10, 4))
    fig.suptitle('Detection Accuracy Overall: 10-40m', size=12)

    axes[0].set_title('Overall Detection Accuracy')
    axes[1].set_title('Test 1 Detection Accuracy')
    axes[2].set_title('Test 2 Detection Accuracy')

    axes[0].set_ylabel('Accuracy %')
    for i in range(columns):
        axes[i].set_ylim([0, 100])
        axes[i].tick_params(axis='x', rotation=0, labelsize=6)
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
                    ha='center', va='top', fontsize=8, color='white')

    # Add labels to bars in each subplot
    add_bar_labels(axes[0], overall)
    add_bar_labels(axes[1], test1_overall)
    add_bar_labels(axes[2], test2_overall)

    plt.tight_layout(pad=1)

    # Save or display the plot
    save = kwargs.get('save', False)
    save_path = kwargs.get('save_path', '')
    if save:
        plt.savefig(f'{save_path}/overall_accuracy.png')
        plt.close(fig)
    else:
        plt.show()


def accuracy_vs_samplength(**kwargs):
    samp_length = [2,3,4,5,6,7,8,9,10]

    experiment = Experiment()
    data = experiment.first_run_df

    # Accuracy By Test ----------------
    spec_70_9k_overall_detection_accuracy_test1 = data[(data['feature'] == 'spectral') & (data['params'] == '70-9000') & (data['test'] == 1)].get(
                'overall')
    spec_350_3k_overall_detection_accuracy_test1 = data[(data['feature'] == 'spectral') & (data['params'] == '350-3000') & (data['test'] == 1)].get(
                'overall')
    mfcc_overall_detection_accuracy_test1 = data[(data['feature'] == 'mfcc') & (data['test'] == 1)].get('overall')

    spec_70_9k_overall_detection_accuracy_test2 = data[(data['feature'] == 'spectral') & (data['params'] == '70-9000') & (data['test'] == 2)].get(
                'overall')
    spec_350_3k_overall_detection_accuracy_test2 = data[(data['feature'] == 'spectral') & (data['params'] == '350-3000') & (data['test'] == 2)].get(
                'overall')
    mfcc_overall_detection_accuracy_test2 = data[(data['feature'] == 'mfcc') & (data['test'] == 2)].get('overall')

    stats = kwargs.get('stats', False)
    if stats:
        print(f'Spec: 70-9k Overall Accuracy: Test 1 - {spec_70_9k_overall_detection_accuracy_test1}')
        print(f'Spec: 350-3k Overall Accuracy: Test 1 - {spec_350_3k_overall_detection_accuracy_test1}')
        print(f'MFCC: 13 Overall Accuracy: Test 1 - {mfcc_overall_detection_accuracy_test1}')
        print('-' * 40)
        print(f'Spec: 70-9k Overall Accuracy: Test 2 - {spec_70_9k_overall_detection_accuracy_test2}')
        print(f'Spec: 350-3k Overall Accuracy: Test 2 - {spec_350_3k_overall_detection_accuracy_test2}')
        print(f'MFCC: 13 Overall Accuracy: Test 2 - {mfcc_overall_detection_accuracy_test2}')
        print('-' * 40)

    colors = ['blue', 'green', 'purple']

    rows = 1
    columns = 2
    gridlines = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    fig, axes = plt.subplots(rows, columns, figsize=(12, 4))
    fig.suptitle('Detection Accuracy vs Sample Length', size=12)

    axes[0].set_title('Test 1')
    axes[1].set_title('Test 2')

    axes[0].set_ylabel('Accuracy %')
    axes[0].set_ylabel('')
    axes[0].set_xlabel('Sample Length (s)')
    axes[1].set_xlabel('Sample Length (s)')
    for i in range(columns):
        axes[i].set_ylim([0, 100])
        axes[i].tick_params(axis='x', rotation=0) #, labelsize=8
        axes[i].set_yticks(gridlines)
        for line in gridlines:
            axes[i].axhline(line, c='gray', linestyle='-', zorder=0, linewidth=0.5, alpha=0.8)

    axes[0].plot(samp_length, spec_70_9k_overall_detection_accuracy_test1, color=colors[0], marker='o', label='Spec: 70-9k')
    axes[0].plot(samp_length, spec_350_3k_overall_detection_accuracy_test1, color=colors[1], marker='o', label='Spec: 350-3k')
    axes[0].plot(samp_length, mfcc_overall_detection_accuracy_test1, color=colors[2], marker='o', label='MFCC: 13')
    axes[1].plot(samp_length, spec_70_9k_overall_detection_accuracy_test2, color=colors[0], marker='o')
    axes[1].plot(samp_length, spec_350_3k_overall_detection_accuracy_test2, color=colors[1], marker='o')
    axes[1].plot(samp_length, mfcc_overall_detection_accuracy_test2, color=colors[2], marker='o')

    legend_lines = [Line2D([0], [0], color=color, lw=2) for color in colors]
    legend_labels = ['Spec: 70-9k', 'Spec: 350-3k', 'MFCC: 13']
    fig.legend(handles=legend_lines, labels=legend_labels, loc='upper center', bbox_to_anchor=(0.5, .07), ncol=3)

    plt.tight_layout(pad=1)

    # Save or display the plot
    save = kwargs.get('save', False)
    save_path = kwargs.get('save_path', '')
    if save:
        plt.savefig(f'{save_path}/overall_accuracy.png')
        plt.close(fig)
    else:
        plt.show()


def model_consistency_worst(**kwargs):
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

    rows = 3
    columns = 2
    gridlines = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    fig, axes = plt.subplots(rows, columns, figsize=(12, 8))
    fig.suptitle('Model Consistency Histogram: MFCC at 7s on Test 1 ', size=20)

    axes[0][0].set_title('Overall Accuracy', size=14)
    axes[0][1].set_title('Negative Samples Accuracy', size=14)
    axes[1][0].set_title('10m Accuracy', size=14)
    axes[1][1].set_title('20m Accuracy', size=14)
    axes[2][0].set_title('30m Accuracy', size=14)
    axes[2][1].set_title('40m Accuracy', size=14)

    axes[0][0].hist(overall, bins=50, range=(0,100), color='#3977b4', edgecolor='#306699')
    axes[0][1].hist(negative, bins=50, range=(0,100), color='#3977b4', edgecolor='#306699')
    axes[1][0].hist(m10, bins=50, range=(0,100), color='#3977b4', edgecolor='#306699')
    axes[1][1].hist(m20, bins=50, range=(0,100), color='#3977b4', edgecolor='#306699')
    axes[2][0].hist(m30, bins=50, range=(0,100), color='#3977b4', edgecolor='#306699')
    axes[2][1].hist(m40, bins=50, range=(0,100), color='#3977b4', edgecolor='#306699')

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
                                     label=f'High Score: {overall_max}%')
    text_legend_2 = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0,
                                     label=f'High Score: {overall_min}%')
    axes[0][0].legend(handles=[line, text_legend_1], loc='upper left')

    plt.tight_layout(pad=1.5)

    # Save or display the plot
    save = kwargs.get('save', False)
    save_path = kwargs.get('save_path', '')
    if save:
        plt.savefig(f'{save_path}/overall_accuracy.png')
        plt.close(fig)
    else:
        plt.show()


def model_consistency_best(**kwargs):
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

    rows = 3
    columns = 2
    gridlines = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    fig, axes = plt.subplots(rows, columns, figsize=(12, 8))
    fig.suptitle('Model Consistency Histogram: Spectral 70-9kHz at 6s on Test 2 ', size=20)

    axes[0][0].set_title('Overall Accuracy', size=14)
    axes[0][1].set_title('Negative Samples Accuracy', size=14)
    axes[1][0].set_title('10m Accuracy', size=14)
    axes[1][1].set_title('20m Accuracy', size=14)
    axes[2][0].set_title('30m Accuracy', size=14)
    axes[2][1].set_title('40m Accuracy', size=14)

    axes[0][0].hist(overall, bins=50, range=(0, 100), color='#3977b4', edgecolor='#306699')
    axes[0][1].hist(negative, bins=50, range=(0, 100), color='#3977b4', edgecolor='#306699')
    axes[1][0].hist(m10, bins=50, range=(0, 100), color='#3977b4', edgecolor='#306699')
    axes[1][1].hist(m20, bins=50, range=(0, 100), color='#3977b4', edgecolor='#306699')
    axes[2][0].hist(m30, bins=50, range=(0, 100), color='#3977b4', edgecolor='#306699')
    axes[2][1].hist(m40, bins=50, range=(0, 100), color='#3977b4', edgecolor='#306699')

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
                                       label=f'High Score: {overall_max}%')
    text_legend_2 = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0,
                                       label=f'High Score: {overall_min}%')
    axes[0][0].legend(handles=[line, text_legend_1], loc='upper left')

    plt.tight_layout(pad=1.5)

    # Save or display the plot
    save = kwargs.get('save', False)
    save_path = kwargs.get('save_path', '')
    if save:
        plt.savefig(f'{save_path}/overall_accuracy.png')
        plt.close(fig)
    else:
        plt.show()


class Experiment:
    def __init__(self):
        base_path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py'
        accuracy_SL_filepath = f'{base_path}/Detection_Classification/Static_3_Exp/graphs/csv/Accuracy-SampLength.csv'
        model_confidence_filepath = f'{base_path}/Detection_Classification/Static_3_Exp/graphs/csv/model-confidence.csv'

        self.first_run_csv = utils.CSVFile(accuracy_SL_filepath)
        self.first_run_df = pd.read_csv(accuracy_SL_filepath)
        self.second_run_csv = utils.CSVFile(model_confidence_filepath)
        self.second_run_df = pd.read_csv(model_confidence_filepath)




if __name__ == '__main__':
    # overall_accuracy(stats=True, save=False)
    # accuracy_vs_samplength(stats=True, save=False)
    # overall_stats(stats=True, save=False)
    # model_consistency_worst(stats=True, save=False)
    model_consistency_best(stats=True, save=False)