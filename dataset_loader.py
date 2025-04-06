from datasets import load_dataset
import pandas as pd

# Description: Load the dataset from the Hugging Face Datasets library
class SpeechDatasetLoader:
    def __init__(self, age_range: tuple = None):
        """Initializes a dataset loader object that loads the huggingface dataset for this project
        Args:
            age_range (tuple): min_age, max_age
        """

        self.dataset = load_dataset("mispeech/speechocean762")
        self.train = self.dataset["train"]  # changed
        self.test = self.dataset["test"]    # changed

        if age_range is not None:
            self.sort_age(age_range)
    
    def sort_age(self, age_range: tuple):
        self.train = self.train.filter(
            lambda example: age_range[0] <= example["age"] < age_range[1]
        )

    def get_dataset(self):
        return self.train, self.test
    
    def get_item(self, item: int):
        return self.train[item]  # changed

if __name__ == "__main__":
    loader = SpeechDatasetLoader((0,8))
    print(loader.get_item(20)["words"][1])