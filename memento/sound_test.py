import sounddevice as sd
import numpy as np

# Define sampling frequency (Hz)
fs = 44100

# Generate a 5-second, 440 Hz sine wave for demonstration
duration = 5.0
t = np.linspace(0., duration, int(duration * fs), dtype='float32')
audio_data = np.sin(2. * np.pi * 440. * t)

# Play the audio
sd.play(audio_data, samplerate=fs)

# Wait until playback is finished
#sd.wait()