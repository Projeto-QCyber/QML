import json
import joblib
import pandas as pd

from typing import Any, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score

class RFModelInput(BaseModel):
    """Input schema for MyCustomTool for predictions using ML model."""
    argument: dict = Field(..., description="The network data sample as a dictionary to be used for prediction.")

class RFModel(BaseTool):
    name: str = "Model"
    description: str = """
        This tool acts as a scikit-learn pre-trained Random Forest model for binary classification tasks.
        It accepts a dictionary of network data features and returns a binary prediction (0 or 1).
    """
    args_schema: Type[BaseModel] = RFModelInput
    model: object

    def __init__(self, model_path: str = None, **kwargs):
        if model_path is None:
            model_path = "D:/Area de trabalho/Faculdade/BioData/Q-Cyber/QML/IA/models/traditional/random_forest_model.joblib"
        
        loaded_model = joblib.load(model_path)

        # Passando o modelo carregado
        super().__init__(model=loaded_model, **kwargs)

    def _run(self, argument: Any) -> str:
        try:
            print("Running RFModel...")
            python_dict = json.loads(argument)
            df_input = pd.DataFrame(python_dict)
            predictions = self.model.predict(df_input)
            print(f"Predictions: {predictions.tolist()}")
            return predictions.tolist()
        except Exception as e:
            return f"Error during prediction: {str(e)}"

"""
if __name__ == "__main__":
    # 1. Instantiate the tool
    # Make sure to have the model file at this path for the test
    model_path_to_test = "D:/Area de trabalho/Faculdade/BioData/Q-Cyber/QML/IA/models/traditional/random_forest_model.joblib"
    rf_tool = RFModel(model_path=model_path_to_test)
    
    # 2. Create a sample input dictionary that matches the expected format
    df = pd.read_csv("D:\\Area de trabalho\\Faculdade\\BioData\\Q-Cyber\\QML\\data\\dados_de_teste.csv").sample(20)
    sample_data_test = df.drop(["Attack_label"], axis=1)
    csv_str = sample_data_test.to_csv(index=False)
    sample_data_test = sample_data_test.reset_index(drop=True)
    dictionary = sample_data_test.to_dict()

    # 3. Call the _run method with the correct input format
    print("Testing the tool with a single sample...")
    result = rf_tool._run(argument=dictionary)
    print(f"Test Result: {result}")
    print(f"Correct predictions: {df['Attack_label'].tolist()}")
    accuracy = accuracy_score(df['Attack_label'].tolist(), result)
    print(f"Accuracy: {accuracy}")
"""