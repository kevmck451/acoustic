import librosa
import librosa.display
import soundfile as sf


# Load your audio file
audio_file = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/5 Bee Project/Data Collection/10-17-24/Synced/Bee Hive 1 chunk 1.wav'
y, sr = librosa.load(audio_file, sr=None)

# Separate harmonic and percussive components
harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 5.0))

# Save the harmonic (non-percussive) part
sf.write('harmonic_only_2.wav', harmonic, sr)
