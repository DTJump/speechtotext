import os
from tkinter import *
from tkinter import messagebox, filedialog
from threading import Thread
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Global variables
device = None
model = None
recording = False
audio = None
stream = None
frames = []
audio_format = None
channels = 1
rate = 16000
chunk = 512

def log(message):
    print(f"{time.strftime('%H:%M:%S')}: {message}")

def load_model():
    global model
    if model is None:
        log("Loading model")
        info_label.config(text="Loading model, please wait...")
        root.update()
        import faster_whisper
        model = faster_whisper.WhisperModel(model_size_or_path="tiny.en", device=device)
        info_label.config(text="Model loaded. Press 'Start Recording' to begin.")
        log("Model loaded")

def start_recording_audio():
    global recording, audio, stream, frames, audio_format
    import pyaudio
    try:
        log("Starting recording")
        recording = True
        audio = pyaudio.PyAudio()
        audio_format = pyaudio.paInt16
        frames = []
        stream = audio.open(rate=rate, format=audio_format, channels=channels, input=True, frames_per_buffer=chunk)
        
        info_label.config(text="Recording... Press 'Stop Recording' to stop.")
        root.update()

        while recording:
            frames.append(stream.read(chunk))
    except Exception as e:
        messagebox.showerror("Error", str(e))
        info_label.config(text="An error occurred.")
        recording = False
        log(f"Error during recording: {e}")

def stop_recording_audio():
    global recording, audio, stream, frames
    import wave
    try:
        log("Stopping recording")
        recording = False
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        info_label.config(text="Processing the audio...")
        root.update()

        with wave.open("voice_record.wav", 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(audio_format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        load_model()

        log("Transcribing audio")
        segments, _ = model.transcribe("voice_record.wav", language="en")
        transcription = " ".join(seg.text for seg in segments)

        transcription_text.delete(1.0, END)
        transcription_text.insert(END, transcription)
        info_label.config(text="Transcription completed.")
        log("Transcription completed")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        info_label.config(text="An error occurred.")
        log(f"Error during processing: {e}")
    finally:
        start_button.config(state=NORMAL)
        stop_button.config(state=DISABLED)

def start_recording():
    start_button.config(state=DISABLED)
    stop_button.config(state=NORMAL)
    Thread(target=start_recording_audio).start()

def save_transcription():
    try:
        log("Saving transcription")
        text = transcription_text.get(1.0, END).strip()
        if not text:
            messagebox.showwarning("Warning", "No transcription to save.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if filepath:
            with open(filepath, 'w') as file:
                file.write(text)
            messagebox.showinfo("Success", "Transcription saved successfully.")
        log("Transcription saved")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        log(f"Error saving transcription: {e}")

# Create the main window
root = Tk()
root.title("Voice-to-Text Transcription")

info_label = Label(root, text="Press 'Start Recording' to begin.")
info_label.pack(pady=10)

start_button = Button(root, text="Start Recording", command=start_recording)
start_button.pack(pady=10)

stop_button = Button(root, text="Stop Recording", command=stop_recording_audio, state=DISABLED)
stop_button.pack(pady=10)

text_frame = Frame(root)
text_frame.pack(pady=10, fill=BOTH, expand=True)

transcription_text = Text(text_frame, wrap=WORD, height=10, width=50)
transcription_text.pack(side=LEFT, fill=BOTH, expand=True)

scrollbar = Scrollbar(text_frame, command=transcription_text.yview)
scrollbar.pack(side=RIGHT, fill=Y)

transcription_text.config(yscrollcommand=scrollbar.set)

save_button = Button(root, text="Save Transcription", command=save_transcription)
save_button.pack(pady=10)

watermark_label = Label(root, text="Made by Andrew Yoon :)", font=("Arial", 8), fg="grey")
watermark_label.pack(side=BOTTOM, pady=5)

# Initialize the device after the main window to prevent lag during startup
def initialize_device():
    global device
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log("Device initialized")

Thread(target=initialize_device).start()

root.mainloop()
