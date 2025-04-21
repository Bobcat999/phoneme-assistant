import tkinter as tk
import json
from phoneme_assistant import PhonemeAssistant

class PhonemeAssistantApp:
    def __init__(self):
        # Initialize the assistant
        self.assistant = PhonemeAssistant()

        # Create the main application window
        self.window = tk.Tk()
        self.window.title("Phoneme Assistant")

        # Sentence input section
        self.sentence_label = tk.Label(self.window, text="Enter a sentence:")
        self.sentence_label.pack()

        self.sentence_entry = tk.Entry(self.window, width=50)
        self.sentence_entry.pack()

        # Run button
        self.run_button = tk.Button(self.window, text="Record & Analyze", command=self.run_assistant)
        self.run_button.pack(pady=10)

        # Feedback display section
        self.output_label = tk.Label(self.window, text="", wraplength=400, justify="left")
        self.output_label.pack(padx=10, pady=10)

        # Word display section (color-coded)
        self.word_display_label = tk.Label(self.window, text="Words and PER (Color-Coded):")
        self.word_display_label.pack()

        self.word_display_text = tk.Text(self.window, wrap="word", width=50, height=5)
        self.word_display_text.pack(padx=10, pady=5)
        self.word_display_text.config(state="disabled")  # Make it read-only

        # Start the Tkinter event loop
        self.window.mainloop()
        pass

    def run_assistant(self):
        attempted_sentence = self.sentence_entry.get().strip()
        if not attempted_sentence:
            self.output_label.config(text="Please enter a sentence.")
            return
        self.output_label.config(text="Recording... Speak now.")
        try:
            # Get the response, DataFrame, and other details from the assistant
            response, df, highest_per_word, problem_summary = self.assistant.record_audio_and_get_response(attempted_sentence, verbose=True)
            output_json = json.loads(response)

            # Display feedback and new sentence
            feedback_text = f"Feedback:\n{output_json['feedback']}\n\nNew Sentence:\n{output_json['sentence']}"
            self.output_label.config(text=feedback_text)

            # Display color-coded words based on PER
            self.display_colored_words(df)

            # Update the sentence entry with the new sentence
            self.sentence_entry.delete(0, tk.END)
            self.sentence_entry.insert(0, output_json["sentence"].replace(".", "").lower())
        except Exception as e:
            self.output_label.config(text=f"Error: {str(e)}")

    def display_colored_words(self, df):
        """
        Display each word from the DataFrame in the text widget, color-coded based on PER.
        """
        self.word_display_text.config(state="normal")
        self.word_display_text.delete("1.0", tk.END)  # Clear previous content

        for _, row in df.iterrows():
            word = row.get("ground_truth_word", "")
            if not word:
                continue
            if not row.get("predicted_word", ""):
                per = 1
            else:
                per = row.get("per", 0)

            red = min(int(per * 255), 255)
            green = max(255 - red, 0)
            color = f"#{red:02x}{green:02x}00"  # RGB to hex

            # Configure the tag for the gradient color
            tag_name = f"color_{color}"
            try:
                # Check if the tag already exists
                self.word_display_text.tag_cget(tag_name, "foreground")
            except tk.TclError:
                # If the tag doesn't exist, define it
                self.word_display_text.tag_config(tag_name, foreground=color)

            # Insert the word with the appropriate tag
            self.word_display_text.insert(tk.END, word + " ", tag_name)
            # Get phonemes and their statuses
            phonemes = row.get("phonemes", [])  # List of phonemes
            statuses = row.get("statuses", [])  # List of statuses (missed, substituted, added)

            # Display phonemes with their statuses
            for phoneme, status in zip(phonemes, statuses):
                if status == "missed":
                    phoneme_color = "red"
                elif status == "substituted":
                    phoneme_color = "orange"
                elif status == "added":
                    phoneme_color = "blue"
                else:
                    phoneme_color = "green"  # Default color for correct phonemes

                # Configure the tag for the phoneme status color
                phoneme_tag = f"phoneme_{phoneme_color}"
                try:
                    self.word_display_text.tag_cget(phoneme_tag, "foreground")
                except tk.TclError:
                    self.word_display_text.tag_config(phoneme_tag, foreground=phoneme_color)

                # Insert the phoneme with the appropriate tag
                self.word_display_text.insert(tk.END, f"{phoneme} ", phoneme_tag)
            self.word_display_text.insert(tk.END, " ")

        self.word_display_text.config(state="disabled")  # Make the text widget read-only
if __name__ == "__main__":
    app = PhonemeAssistantApp()