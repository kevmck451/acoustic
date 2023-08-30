import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

# === Generating Test Data ===
# Look at model accuracy as a mic system
# directory = directory_test_3
# truth = generate_truth(directory_test_3)
# overall_accuracy, filenames, scores = test_model_accuracy(model, directory, truth, display=False)
# np.save('test_3_filenames.npy', filenames)
# np.save('test_3_scores.npy', scores)

# === Data Loading ===
names, scores = np.load('accuracy/test_3_filenames.npy'), np.load(
    'accuracy/test_3_scores.npy')

# === Data Preparation ===
# Filter out the 'hover' entries and their corresponding scores
free_flight_indices = [i for i, name in enumerate(names) if 'hover' not in name]
free_flight_names = names[free_flight_indices]
free_flight_scores = scores[free_flight_indices]

# === Data Processing ===
# Create a dictionary where the keys are the systems, and the values are lists of scores for that system
systems_scores = {}
for name, score in zip(free_flight_names, free_flight_scores):
    system = name.rsplit('_', 1)[0]
    if system not in systems_scores:
        systems_scores[system] = []
    systems_scores[system].append(score)

# Compute the average score for each system
average_scores = {system: np.mean(scores) for system, scores in systems_scores.items()}

# Extract the number from the string for sorting
def get_key(item):
    number = int(re.search(r'\d+', item[0]).group())
    return number

# Sort the average_scores dictionary by the extracted number
sorted_average_scores = sorted(average_scores.items(), key=get_key)

# === Plotting ===
# Create DataFrame
data = pd.DataFrame(sorted_average_scores, columns=['System', 'Score'])

# Calculate accuracy
accuracy = int(np.round(len(data[data['Score'] >= 50]) / len(data) * 100))

# Create subplots
fig, ax = plt.subplots(figsize=(14, 6))

# Plot the scores
bars = ax.bar(data['System'], data['Score'], color=data['Score'].apply(lambda x: 'g' if x >= 50 else 'r'))

# Draw a line at 50% score
ax.axhline(50, color='black', linestyle='dotted', label='Flight_Analysis Threshold')

# Set title and labels
ax.set_title(f'Free Flight System Predictions: {accuracy}%', size=14)
ax.set_xlabel('System')
ax.set_ylabel('Score (%)')

# Add score numbers on top of each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval - 5, int(yval), ha='center', va='bottom', color='white', fontsize=8)

# Set x-axis rotation
ax.tick_params(axis='x', rotation=90)

# Create custom legend handles and labels
legend_handles = [
    mpatches.Patch(color='g', label='Predicted Correctly'),
    mpatches.Patch(color='r', label='Predicted Incorrect'),
    mpatches.Patch(color='black', linestyle='dotted', label='Flight_Analysis Threshold')]

ax.legend(loc='upper left', handles=legend_handles)

# Adjust subplot parameters to give specified padding
plt.tight_layout(pad=1)

plt.show()
