import tkinter as tk
from tkinter import filedialog, scrolledtext
import os
import whisper
import warnings
import subprocess  # Import the subprocess module for FFmpeg conversion
from googletrans import Translator, LANGUAGES
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")
model_size = "medium"
model = WhisperModel(model_size,compute_type="int8")

def translate_text(text, target_language):
    translator = Translator()
    
    # Detect the source language (optional)
    detected_language = translator.detect(text).lang
    
    # Translate the text to the target language
    translated_text = translator.translate(text, src=detected_language, dest=target_language)
    
    return translated_text.text

def browse():
    global file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        print("Selected file:", file_path)

def convert_audio():
    global file_path
    if not file_path:
        print("Please select an audio file first.")
        return


    # Load and process the converted audio
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    # Detect the spoken language
    segments, info = model.transcribe(file_path, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    
    # Concatenate segments into a single string
    recognized_text = " ".join([segment.text for segment in segments])
    
    if recognized_text:
        # Translate the recognized text
        translated_text = translate_text(recognized_text, 'en')

        # Update the text widgets
        original_text_widget.delete(2.0, tk.END)
        original_text_widget.insert(tk.END, f"Detected Language: {info.language}\n\n{recognized_text}")

        translated_text_widget.delete(2.0, tk.END)
        translated_text_widget.insert(tk.END, f"Translated Text:\n\n{translated_text}")
    else:
        print("No recognized text to translate.")



root = tk.Tk()
root.title("Audio Transcription and Analysis")

browse_button = tk.Button(root, text="Browse", command=browse)
browse_button.pack(pady=20)

convert_button = tk.Button(root, text="Convert and Transcribe Audio", command=convert_audio)
convert_button.pack(pady=20)

# Create text widgets for displaying original and translated text
original_text_widget = scrolledtext.ScrolledText(root, width=40, height=20)
original_text_widget.pack(pady=20, padx=20, side=tk.LEFT)

translated_text_widget = scrolledtext.ScrolledText(root, width=40, height=20)
translated_text_widget.pack(pady=20, padx=20, side=tk.RIGHT)

root.mainloop()
