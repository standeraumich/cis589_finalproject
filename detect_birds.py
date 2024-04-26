import numpy as np
import pyaudio
import time
import geocoder
import wave
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
import os
import pandas as pd
import requests 

url = 'http://localhost:3001/birds'
g = geocoder.ip('me')
# Parameters for audio recording
RATE = 44100  # Sampling rate
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Number of audio channels (mono)

full_data = []


# Initialize PyAudio
p = pyaudio.PyAudio()

# Open an audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Initialize variables for spectrogram
window = np.blackman(CHUNK)
max_freq = 0
max_power = 0

# Initialize variables for noise floor calculation
noise_floor_samples = []
noise_floor_duration = 10 # seconds
noise_floor_threshold = 0  # Initialize with zero (will be updated)

# Record audio for noise floor calculation
print("Calculating noise floor for 10 seconds...")
start_time = time.time()
while time.time() - start_time < noise_floor_duration:
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.int16)
    noise_floor_samples.extend(audio_data)

# Compute the FFT of the audio data
noise_fft_data = np.fft.fft(noise_floor_samples)    
noise_freqs = np.fft.fftfreq(len(noise_floor_samples), 1 / RATE)
noise_power_spectrum = np.abs(noise_fft_data) ** 2

# Find the frequency with the highest power
noise_max_power_index = np.argmax(noise_power_spectrum)
noise_max_freq = noise_freqs[noise_max_power_index]
noise_max_power = noise_power_spectrum[noise_max_power_index]
noise_floor_threshold = noise_max_power / 4

print(f"Noise floor threshold: {noise_floor_threshold:.2f}")

output_filename_template = "audio_samples_{count}.wav"
output_wavefile = None
file_count = 0
in_max_power = False
# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()

buffer = 100


try:
    while True:
        # Read audio data from the stream
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Compute the FFT of the audio data
        fft_data = np.fft.fft(audio_data * window)
        freqs = np.fft.fftfreq(CHUNK, 1 / RATE)
        power_spectrum = np.abs(fft_data) ** 2

        # Find the frequency with the highest power
        max_power_index = np.argmax(power_spectrum)
        max_freq = freqs[max_power_index]
        max_power = power_spectrum[max_power_index]

        # Print the result if above noise floor threshold
        if ((max_power > noise_floor_threshold) and (max_freq >= 1000 and max_freq <= 8000)) or (buffer < 150):
            if in_max_power == False:
                buffer = 0
                in_max_power = True
            if max_power > noise_floor_threshold and (max_freq >= 1000 and max_freq <= 8000):
                buffer = 0
            # print(f"Frequency with highest power: {max_freq:.2f} Hz (Power: {max_power:.2f})")
            # print('recording')
            print(buffer)
             # Save audio samples to the output file
            full_data.extend(audio_data)
            buffer+=1
            
        else:
            in_max_power = False
            full_data = np.array(full_data)
            # If currently writing to a file, close it
            duration = full_data.size / RATE
            if duration > 3 and output_wavefile is None:
                print(duration)
                file_count += 1
                output_filename = output_filename_template.format(count=file_count)
                output_wavefile = wave.open(output_filename, 'wb')
                output_wavefile.setnchannels(CHANNELS)
                output_wavefile.setsampwidth(p.get_sample_size(FORMAT))
                output_wavefile.setframerate(RATE)
                output_wavefile.writeframes(full_data.tobytes())
                output_wavefile.close()
                print(f"Audio samples saved to {output_filename}")
                recording = Recording(
                    analyzer,
                    output_filename,
                    lat=g.latlng[0],
                    lon=g.latlng[1],
                    date=datetime.now(),
                    min_conf=0.25,
                    )
                recording.analyze()
                birds=[]
                for item in recording.detections:
                    birds.append(item['common_name'])
                birds = list(set(birds))
                print(birds)
                for bird in birds: 
                    data_obj = {"bird": bird, "lat":g.latlng[0], "lon":g.latlng[1]}
                    x = requests.post(url, json = data_obj)
                    print(x.text)
                # os.remove(output_filename)
            output_wavefile = None
            full_data = []

except KeyboardInterrupt:
    print("Recording stopped by user.")
finally:
    # Clean up
    if output_wavefile is not None:
        output_wavefile.close()
        print(f"Audio samples saved to {output_filename}")
    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()
