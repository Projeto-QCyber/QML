import os
import pandas as pd

from qml.crew import CyberPredict

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

def run():
    """
    Run the research crew.
    """
    
    data_path = "./data/dados_de_teste.csv"
    """
    data = pd.read_csv(data_path).sample(10)
    sample_data_test = data.drop(["Attack_label"], axis=1)
    sample_data_test = sample_data_test.reset_index(drop=True)
    dictionary = sample_data_test.to_dict()
    inputs = {
        'argument': dictionary
    }
    """

    result = CyberPredict().crew().kickoff(inputs=data_path)

    print("\n\n=== FINAL RESULT ===\n\n")
    print("Predictions:")
    print(result.predictions)
    print("Report:")
    print(result.report)


if __name__ == "__main__":
    run()