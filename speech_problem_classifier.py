from collections import Counter

class SpeechProblemClassifier:
    """
    Analyzes speech analysis results to detect the most common phoneme or word-level problems.
    """

    @staticmethod
    def classify_problems(results):
        """
        Analyzes the results to find the most common phoneme and word-level problems.
        
        Args:
            results (list): List of dictionaries containing phoneme and word-level analysis. produced by process_audio
        
        Returns:
            dict: A summary of the most common problems.
        """
        phoneme_errors = Counter()
        word_errors = Counter()

        for result in results:
            if result["type"] in ("match", "substitution"):
                # Count missed, added, and substituted phonemes
                phoneme_errors.update(result.get("missed", []))
                phoneme_errors.update(result.get("added", []))
                phoneme_errors.update([sub[0] for sub in result.get("substituted", [])])  # Incorrect phonemes

            elif result["type"] == "deletion":
                # Count missing words
                word_errors[result["ground_truth_word"]] += 1

            elif result["type"] == "insertion":
                # Count extra predicted words
                word_errors[result["predicted_word"]] += 1

        # Find the most common phoneme and word errors
        most_common_phoneme = phoneme_errors.most_common(1)
        most_common_word = word_errors.most_common(1)

        return {
            "most_common_phoneme": most_common_phoneme[0] if most_common_phoneme else None,
            "most_common_word": most_common_word[0] if most_common_word else None,
            "phoneme_error_counts": phoneme_errors,
            "word_error_counts": word_errors,
        }