import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the confusion matrix data
file_path = '/home/zhang/Desktop/Code/Other/epoch_49_test_confusion_matrix.json'
with open(file_path, 'r') as file:
    confusion_matrix_data = json.load(file)

# Define the mapping from activity to its corresponding index
elderly_code_labels = {
    1: "eating food with a fork",
    2: "pouring water into a cup",
    3: "taking medicine",
    4: "drinking water",
    5: "putting food in the fridge/taking food from the fridge",
    6: "trimming vegetables",
    7: "peeling fruit",
    8: "using a gas stove",
    9: "cutting vegetable on the cutting board",
    10: "brushing teeth",
    11: "washing hands",
    12: "washing face",
    13: "wiping face with a towel",
    14: "putting on cosmetics",
    15: "putting on lipstick",
    16: "brushing hair",
    17: "blow drying hair",
    18: "putting on a jacket",
    19: "taking off a jacket",
    20: "putting on/taking off shoes",
    21: "putting on/taking off glasses",
    22: "washing the dishes",
    23: "vacuumming the floor",
    24: "scrubbing the floor with a rag",
    25: "wipping off the dinning table",
    26: "rubbing up furniture",
    27: "spreading bedding/folding bedding",
    28: "washing a towel by hands",
    29: "hanging out laundry",
    30: "looking around for something",
    31: "using a remote control",
    32: "reading a book",
    33: "reading a newspaper",
    34: "handwriting",
    35: "talking on the phone",
    36: "playing with a mobile phone",
    37: "using a computer",
    38: "smoking",
    39: "clapping",
    40: "rubbing face with hands",
    41: "doing freehand exercise",
    42: "doing neck roll exercise",
    43: "massaging a shoulder oneself",
    44: "taking a bow",
    45: "talking to each other",
    46: "handshaking",
    47: "hugging each other",
    48: "fighting each other",
    49: "waving a hand",
    50: "flapping a hand up and down (beckoning)",
    51: "pointing with a finger",
    52: "opening the door and walking in",
    53: "fallen on the floor",
    54: "sitting up/standing up",
    55: "lying downn"
}

# Initialize an empty confusion matrix
n_labels = len(elderly_code_labels)
confusion_matrix = np.zeros((n_labels, n_labels))

# Populate the confusion matrix
# Assuming each activity in confusion_matrix_data corresponds directly to the index in elderly_code_labels
for activity, data in confusion_matrix_data.items():
    # The index for true labels is the position in the elderly_code_labels, adjusted for 0-based indexing
    true_index = list(elderly_code_labels.values()).index(activity)
    for prediction_data in data[:-1]:  # Exclude the last item which is the total count
        predicted_activity, count = prediction_data.rsplit('  ', 1)
        predicted_index = list(elderly_code_labels.values()).index(predicted_activity)  # Adjust for 0-based indexing
        confusion_matrix[true_index, predicted_index] = int(count)

# Define labels for the plot based on the order in elderly_code_labels
labels = list(range(1, n_labels + 1))

# # Reverse the mapping to get activity names by index
# index_to_activity = {v: k for k, v in elderly_code_labels.items()}

# # Initialize an empty confusion matrix
# n_labels = len(elderly_code_labels)
# confusion_matrix = np.zeros((n_labels, n_labels))

# # Populate the confusion matrix
# for activity, data in confusion_matrix_data.items():
#     # Extract the true label index
#     true_index = index_to_activity[activity] - 1  # Adjust for 0-based indexing
#     for prediction_data in data[:-1]:  # Exclude the last item which is the total count
#         predicted_activity, count = prediction_data.rsplit('  ', 1)
#         predicted_index = index_to_activity[predicted_activity] - 1  # Adjust for 0-based indexing
#         confusion_matrix[true_index, predicted_index] = int(count)

# Plot the confusion matrix
plt.figure(figsize=(20, 20))
#sns.heatmap(confusion_matrix, annot=True, cmap="Blues", xticklabels=elderly_code_labels.values(), yticklabels=elderly_code_labels.values())
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
#plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('Baseline_confusion_matrix.png')
plt.close()
#plt.show()
