#%%
from dataset_loader import SpeechDatasetLoader
from phoneme_extractor import PhonemeExtractor
from word_extractor import WordExtractor
from process_audio import process_audio_array, analyze_results
from grapheme_to_phoneme import grapheme_to_phoneme
import pandas as pd
import math
import matplotlib.pyplot as plt  # changed

#%%
loader = SpeechDatasetLoader((0,10), True)
phoneme_model = PhonemeExtractor()
word_model = WordExtractor()
#%%

# ACCURACY IS MEASURED AS THE CORRELATION COEFFICIENT BETWEEN THE DATABASE ACCURACY AND THE MODEL PER

class MeasureAccuracy:
    def __init__(self, dataset = SpeechDatasetLoader((0,10)), phoneme_model = PhonemeExtractor(), word_model = WordExtractor()):
        self.dataset = dataset
        self.phoneme_model = phoneme_model
        self.word_model = word_model

    def compare_indexes(self, index_range: range, merged_list=None) -> tuple[pd.DataFrame, float, float, float]:
        if merged_list is None:
            all_results = []
            for idx in index_range:
                try:
                    df, _ = self.compare_index(idx)
                    all_results.append(df)
                except Exception as e:
                    print(f"Skipping index {idx} due to error: {e}")

            if not all_results:
                return pd.DataFrame(), 0.0, 0.0, 0.0

            merged = pd.concat(all_results, ignore_index=True)
        else:
            merged = merged_list

        model_per_vals = pd.Series(merged["model_per"]).fillna(1).to_list()
        gt_per_vals = pd.Series(merged["ground_truth_per"]).fillna(1).to_list()

        # mean absolute error
        mae = sum(abs(m - g) for m, g in zip(model_per_vals, gt_per_vals)) / len(model_per_vals)

        # mean squared error
        mse = sum((m - g) ** 2 for m, g in zip(model_per_vals, gt_per_vals)) / len(model_per_vals)

        # correlation coefficient
        corr = self.correlation_coefficient(model_per_vals, gt_per_vals)

        return merged, mae, mse, corr

    def compare_index(self, index: int) -> tuple[pd.DataFrame, float]:
        """Function that compares our model to the  

        Args:
            index (int): _description_

        Returns:
            tuple[pd.DataFrame, float]: DataFrame and correlation coefficient
        """
        # gets the item from our dataset
        item = self.dataset.get_item(index)

        # gets the audio
        audio_array = item["audio"]["array"]

        # the text we were supposed to say
        text = item["text"]
        ground_truth = grapheme_to_phoneme(text.lower()) 

        # use our model to process and analyze our results
        results = process_audio_array(ground_truth, audio_array, 16000, phoneme_extraction_model=self.phoneme_model, word_extraction_model=self.word_model)
        df, _, _ = analyze_results(results)

        # construct our output dataframe
        results = []

        # compare our resulting per per word to that of the database
        word_info = item.get("words", [])
        clean_word_info = self.clean_dataset_data(word_info)  # changed
        per = []
        if word_info:
            for i, w in enumerate(clean_word_info):
                ground_truth_per = len(w["mispronunciations"]) / len(w["phones"])
                if ground_truth_per == 0 or w["accuracy"] == 10:
                    continue  # Skip words with ground_truth_per == 0 or accuracy == 10
                print("Word info:", w)
                print("Model info:", df.iloc[i])
                model_per = df.iloc[i]["per"]
                results.append({
                    "word": w["text"],
                    "model_per": model_per,
                    "ground_truth_accuracy": w["accuracy"],
                    "ground_truth_per": ground_truth_per,
                })

        # turn results into a dataframe
        results = pd.DataFrame(results)

        # figure out correlation coef of accuracy and per
        correlation_coefficient = self.correlation_coefficient(results["model_per"].to_list(), results["ground_truth_accuracy"].to_list())

        return results, correlation_coefficient

    def correlation_coefficient(self, x: list, y: list) -> float:
        """Finds the correlation coefficient between two same sized lists

        Args:
            x (list): the x part of the data
            y (list): the y part of the data

        Returns:
            float: the correlation coefficient between the two
        """
        if len(x) != len(y):
            raise AttributeError("The length of x and y must be the same")

        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denominator = math.sqrt(
            sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)
        )
        if denominator == 0:
            return 0.0

        return numerator / denominator

    def clean_dataset_data(self, data: list) -> list:
        """Converts dataset data into a usable form

        Args:
            data (list): dataframe with a format similar to our processed audio
        """
        # data is a list of word-level dictionaries
        records = []
        for entry in data:
            records.append({
                "text": entry["text"],
                "accuracy": entry["accuracy"],
                "phones": entry["phones"],
                "mispronunciations": entry["mispronunciations"],
            })
        return records
    
#%%
accuracy = MeasureAccuracy(dataset=loader, phoneme_model=phoneme_model, word_model=word_model)
results, correlation_coefficient = accuracy.compare_index(300)
print(results)
print("correlation coef:", correlation_coefficient)
#%%
import numpy as np
accuracy = MeasureAccuracy(dataset=loader, phoneme_model=phoneme_model, word_model=word_model)
results = accuracy.compare_indexes(map(int,list(np.random.randint(1,len(loader.get_dataset()[0]),50))))
#%%
print(results)
print("correlation coef:", results[3])
print("mean squared error", results[2])
print("mean absolute error", results[1])

# Import necessary libraries
import numpy as np

# Filter out invalid data for ground_truth_accuracy vs model_per
valid_data = results[0].dropna(subset=["ground_truth_accuracy", "model_per"])
x = valid_data["ground_truth_accuracy"]
y = valid_data["model_per"]

# Plot ground_truth_accuracy vs model_per with extended line of best fit
plt.scatter(x, y, label="Data Points")
coefficients = np.polyfit(x, y, 1)  # Linear fit (degree 1)
line = np.poly1d(coefficients)

# Define the range of x values to cover the entire graph
x_range = np.linspace(0, 10, 100)  # From 0 to 10 with 100 points
plt.plot(x_range, line(x_range), color="red", label="Best Fit Line")  # Extended line]
plt.xlabel("Ground Truth Accuracy")
plt.ylabel("Model PER")
plt.xlim(0, 10)  # Ensure the x-axis goes from 0 to 10
plt.legend()
plt.title("Ground Truth Accuracy vs Model PER")
plt.show()

# Filter out invalid data for ground_truth_per vs model_per
valid_data = results[0].dropna(subset=["ground_truth_per", "model_per"])
x = valid_data["ground_truth_per"]
y = valid_data["model_per"]

# Plot ground_truth_per vs model_per with extended line of best fit
plt.scatter(x, y, label="Data Points")
coefficients = np.polyfit(x, y, 1)  # Linear fit (degree 1)
line = np.poly1d(coefficients)

# Define the range of x values to cover the entire graph
x_range = np.linspace(0, 1, 100)  # From 0 to 1 with 100 points
plt.plot(x_range, line(x_range), color="red", label="Best Fit Line")  # Extended line
plt.xlabel("Ground Truth PER")
plt.ylabel("Model PER")
plt.xlim(0, 1)  # Ensure the x-axis goes from 0 to 1
plt.legend()
plt.title("Ground Truth PER vs Model PER")
plt.show()