import csv
import os
import eng_to_ipa
from evaluation.accuracy_metrics import compute_phoneme_error_rate

def clean_text(text: str) -> str:
    """
    Removes periods and apostrophes from the text.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text without periods and apostrophes.
    """
    return text.replace('.', '').replace("ˈ", '')

def clean_phonemes(phonemes: str) -> str:
    """
    Removes asterisks from the phoneme output.

    Args:
        phonemes (str): Input phoneme string.

    Returns:
        str: Cleaned phoneme string.
    """
    return phonemes.replace('*', '')

def read_original_sentences(csv_filepath: str) -> list[tuple[str, str]]:
    """
    Reads original and altered sentences from a CSV file.

    Args:
        csv_filepath (str): Path to the CSV file.

    Returns:
        list[tuple[str, str]]: List of tuples containing original and altered sentences.
    """
    sentences = []
    with open(csv_filepath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Ensure the row is not empty
                sentences.append((row[0], row[1]))  # Assuming original and altered sentences are in the first two columns
    return sentences

def write_phoneme_sentences(input_csv: str, output_csv: str):
    """
    Converts sentences to phonemes, calculates PER, and writes them to a new CSV file.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
    """
    sentences = read_original_sentences(input_csv)
    with open(output_csv, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Original Sentence", "Altered Sentence", "Original Phonemes", "Altered Phonemes", "PER"])  # Header row
        for idx, (original, altered) in enumerate(sentences[2:], start=1):
            original_cleaned = clean_text(original)
            altered_cleaned = clean_text(altered)
            original_phonemes = clean_phonemes(eng_to_ipa.convert(original_cleaned))
            altered_phonemes = clean_phonemes(eng_to_ipa.convert(altered_cleaned))
            per = compute_phoneme_error_rate(original_phonemes.split(), altered_phonemes.split())
            writer.writerow([f"{idx}", original_cleaned, altered_cleaned, original_phonemes, altered_phonemes, per])

if __name__ == "__main__":
    input_csv_path = os.path.join(os.getcwd(), "dataset/originalscentences.csv")
    output_csv_path = os.path.join(os.getcwd(), "dataset/phoneme_sentences.csv")
    write_phoneme_sentences(input_csv_path, output_csv_path)
    print(f"Phoneme sentences written to {output_csv_path}")