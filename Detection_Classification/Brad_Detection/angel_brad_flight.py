




from Flight_Analysis_Old.Flight_Path.flight_path import Flight_Path
from Flight_Analysis_Old.Targets.target import Target

from sklearn.metrics import accuracy_score
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


'''
Pass 1: 1m 40.654375s
Pass 2: 2m 19.609375s
Pass 3: 2m 58.486375s
Pass 4: 3m 38.282375s
Pass 5: 4m 17.198375s
Pass 6: 4m 57.234375s
Pass 7: 5m 36.710375s
Pass 8: 6m 15.146375s
Pass 9: 6m 53.102375s
'''


def brad_accuracy(predicted):
    labels = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]

    neg_total = len([x for x in labels if x == 0])
    pos_total = len([x for x in labels if x == 1])
    pos_id = 0
    neg_id = 0

    for p, l in zip(predicted, labels):
        if l == 0 and p == 0: neg_id+=1
        elif l == 1 and p == 1: pos_id+=1

    neg_score = int(np.round((neg_id/neg_total), 2) * 100)
    pos_score = int(np.round((pos_id/pos_total), 2) * 100)

    # Compute accuracy
    accuracy = accuracy_score(labels, predicted)
    accuracy = int(np.round((accuracy * 100)))

    print(f'Threat Accuracy: {pos_score}%')
    print(f'No Threat Accuracy: {neg_score}%')
    print(f'Overall Accuracy: {accuracy}%')

    # Create DataFrame
    data = pd.DataFrame({
        'Label': labels,
        'Predicted': predicted
    })

    # Separate negatives and positives
    negatives = data[data['Label'] == 0]
    positives = data[data['Label'] == 1]

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(f'Angel_1_Brad Accuracy: {accuracy}%', size=14)

    # Plot negatives
    axes[0].bar(negatives.index, negatives['Predicted'],
                color=negatives.apply(lambda row: 'g' if row['Label'] == row['Predicted'] else 'r', axis=1))
    axes[0].set_ylim([0, 1])  # Set y-axis limits for binary prediction
    axes[0].set_title(f'Negatives: {neg_score}%')
    axes[0].set_ylabel('Predicted')
    axes[0].set_xticks(negatives.index)

    # Create custom legend handles and labels
    legend_handles = [
        mpatches.Patch(color='g', label='Predicted Correctly'),
        mpatches.Patch(color='r', label='Predicted Incorrect')]

    axes[0].legend(loc='upper left', handles=legend_handles)

    # Plot positives
    axes[1].bar(positives.index, positives['Label'],
                color=positives['Predicted'].apply(lambda x: 'g' if x == 1 else 'r'))
    axes[1].set_ylim([0, 1])  # Set y-axis limits for percentage
    axes[1].set_xlabel('Samples')
    axes[1].set_ylabel('Label')
    axes[1].set_title(f'Positives: {pos_score}%')
    axes[1].set_xticks(positives.index)

    plt.tight_layout(pad=1)  # Adjust subplot parameters to give specified padding
    plt.show()



if __name__ == '__main__':

    target = Target(name='Semi', type='speaker', flight='Angel_1')
    flight = Flight_Path('Angel_1', target_object=target)  #

    # flight.plot_flight_path(save=True)
    flight.display_target_distance(display=True)
    # flight.get_takeoff_time(display=True)



    all_flight_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    all_flight_2 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    all_flight_3 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    all_flight_3n = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    all_flight_4 = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    all_flight_5n = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    all_flight_5 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                    0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0]

    full_list = [all_flight_1, all_flight_2, all_flight_3, all_flight_3n, all_flight_4, all_flight_5n, all_flight_5]

    # for predicted in full_list:
    #     brad_accuracy(predicted)