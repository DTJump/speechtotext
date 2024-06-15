import os
os.environ ['KMP_DUPLICATE_LIB_OK']='True'
import pyaudio
import wave
import keyboard
import faster_whisper
import torch.cuda

# Initialize Whisper model
model = faster_whisper.WhisperModel(model_size_or_path="tiny.en", device='cuda')

def transcribe_audio():
    # Wait until user presses space bar
    print("\n\nTap space when you're ready. ", end="", flush=True)
    keyboard.wait('space')
    while keyboard.is_pressed('space'): pass

    # Record from microphone until user presses space bar again
    print("Tap space when you're done.\n")
    audio = pyaudio.PyAudio()
    frames = []
    stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    while not keyboard.is_pressed('space'):
        frames.append(stream.read(512))
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a WAV file
    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))

    # Transcribe the recording using Whisper
    transcription = " ".join(seg.text for seg in model.transcribe("voice_record.wav", language="en")[0])
    print(f'Transcription: {transcription}')

if __name__ == "__main__":
    while True:
        transcribe_audio()