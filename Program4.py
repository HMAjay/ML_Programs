#Find-S Algorithm for Hypothesis Generation
import pandas as pd
import numpy as np

# Function to implement Find-S algorithm
def find_s_algorithm(training_data):

    hypothesis = None
    temp = None   # To store last-but-one hypothesis

    # Iterate through training examples
    for index, row in training_data.iterrows():

        # Check if positive example
        if row.iloc[-1].lower() == "yes":

            if hypothesis is None:
                # Initialize with first positive example
                hypothesis = row.iloc[:-1].values.copy()

            else:
                # Save previous hypothesis
                temp = hypothesis.copy()

                # Generalize hypothesis
                for i in range(len(hypothesis)):
                    if hypothesis[i] != row.iloc[i]:
                        hypothesis[i] = "?"

    # If hypothesis becomes all '?', return previous one
    if hypothesis is not None:
        if len(hypothesis) == list(hypothesis).count("?"):
            return temp

    return hypothesis


# Load dataset from CSV file
csv_file = "training_data.csv"

df = pd.read_csv(csv_file)

# Display dataset
print("Training Data:\n", df)

# Run Find-S algorithm
final_hypothesis = find_s_algorithm(df)

# Print final hypothesis
print("\nFinal Hypothesis:", final_hypothesis)