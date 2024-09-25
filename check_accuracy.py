import pandas as pd

def calculate_accuracy(correct_csv, predicted_csv):
    # Load the correct labels and predicted labels from CSV files
    correct_df = pd.read_csv(correct_csv)
    predicted_df = pd.read_csv(predicted_csv)
    
    # Ensure both CSVs have the same structure and are properly aligned
    if len(correct_df) != len(predicted_df):
        raise ValueError("The number of rows in the correct and predicted CSV files do not match.")

    # Ensure the files contain the same image filenames in the same order
    if not all(correct_df['filename'] == predicted_df['filename']):
        raise ValueError("The filenames in the correct and predicted CSV files do not match or are out of order.")
    
    # Calculate accuracy
    correct_predictions = (correct_df['label'] == predicted_df['label']).sum()
    total_predictions = len(correct_df)
    accuracy = correct_predictions / total_predictions
    
    print(f"Prediction Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    # Example usage
    correct_csv_path = "./hw1_data/p1_data/office/train.csv"  # Replace with the path to the correct labels CSV file
    predicted_csv_path = "./haha.csv"  # Replace with the path to the predicted labels CSV file
    
    calculate_accuracy(correct_csv_path, predicted_csv_path)
