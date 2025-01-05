import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).
    """
    evidence = []
    labels = []

    with open(filename) as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            # Extract evidence (features)
            evidence.append([
                int(row[0]),  # Administrative
                float(row[1]),  # Administrative_Duration
                int(row[2]),  # Informational
                float(row[3]),  # Informational_Duration
                int(row[4]),  # ProductRelated
                float(row[5]),  # ProductRelated_Duration
                float(row[6]),  # BounceRates
                float(row[7]),  # ExitRates
                float(row[8]),  # PageValues
                float(row[9]),  # SpecialDay
                month_to_int(row[10]),  # Month (0-11)
                int(row[11]),  # OperatingSystems
                int(row[12]),  # Browser
                int(row[13]),  # Region
                int(row[14]),  # TrafficType
                int(row[15] == 'Returning'),  # VisitorType (1 for Returning, 0 for New)
                int(row[16] == 'TRUE')  # Weekend (1 if TRUE, 0 if FALSE)
            ])

            # Extract label (whether they made a purchase)
            labels.append(1 if row[17] == 'TRUE' else 0)

    return evidence, labels

def month_to_int(month_str):
    """Convert month name to corresponding integer index (0 to 11)."""
    months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    # Normalize month string (capitalize first letter and lowercase the rest)
    month_str = month_str.strip().capitalize()

    try:
        return months.index(month_str)
    except ValueError:
        print(f"Unexpected month: {month_str}")
        return -1  # Or any appropriate value for error handling

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).
    """
    true_positives = sum((labels[i] == 1 and predictions[i] == 1) for i in range(len(labels)))
    true_negatives = sum((labels[i] == 0 and predictions[i] == 0) for i in range(len(labels)))
    false_positives = sum((labels[i] == 0 and predictions[i] == 1) for i in range(len(labels)))
    false_negatives = sum((labels[i] == 1 and predictions[i] == 0) for i in range(len(labels)))

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    return sensitivity, specificity

if __name__ == "__main__":
    main()

