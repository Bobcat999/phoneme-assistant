from datasets import load_dataset

# Description: Load the dataset from the Hugging Face Datasets library
class SpeechDatasetLoader:
    def __init__(self):
        self.dataset = load_dataset("mispeech/speechocean762")
        self.train = self.dataset["train"]
        self.test = self.dataset["test"]

    def get_dataset(self):
        return self.train, self.test
