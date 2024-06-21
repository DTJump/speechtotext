import os
os.environ ['KMP_DUPLICATE_LIB_OK']='True'
import pyaudio
import wave
import keyboard
import faster_whisper
import torch.cuda
from tkinter import *
from tkinter import messagebox

# Initialize Whisper model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = faster_whisper.WhisperModel(model_size_or_path="tiny.en", device=device)

def transcribe_audio():
    try:
        # Disable the start button to prevent multiple recordings
        start_button.config(state=DISABLED)
        
        # Update the GUI to show waiting message
        info_label.config(text="Tap space when you're ready...")
        root.update()
        keyboard.wait('space')
        while keyboard.is_pressed('space'): pass

        # Update the GUI to show recording message
        info_label.config(text="Recording... Tap space when you're done.")
        root.update()
        audio = pyaudio.PyAudio()
        frames = []
        stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
        while not keyboard.is_pressed('space'):
            frames.append(stream.read(512))
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Update the GUI to show processing message
        info_label.config(text="Processing the audio...")
        root.update()

        # Save the recorded audio to a WAV file
        with wave.open("voice_record.wav", 'wb') as wf:
            wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
            wf.writeframes(b''.join(frames))

        # Transcribe the recording using Whisper
        segments, _ = model.transcribe("voice_record.wav", language="en")
        transcription = " ".join(seg.text for seg in segments)
        
        # Update the transcription text box and info label
        transcription_text.delete(1.0, END)
        transcription_text.insert(END, transcription)
        info_label.config(text="Transcription completed.")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        info_label.config(text="An error occurred.")
    finally:
        # Re-enable the start button after processing is complete
        start_button.config(state=NORMAL)

# Create the main window
root = Tk()
root.title("Voice-to-Text Transcription")

# Create and place widgets
info_label = Label(root, text="Press 'Start Recording' to begin.")
info_label.pack(pady=10)

start_button = Button(root, text="Start Recording", command=transcribe_audio)
start_button.pack(pady=10)

transcription_text = Text(root, wrap=WORD, height=10, width=50)
transcription_text.pack(pady=10)

# Run the main loop
root.mainloop()