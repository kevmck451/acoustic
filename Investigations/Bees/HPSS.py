import librosa
import librosa.display
import soundfile as sf


# Load your audio file
filename = 'bees'
# audio_file = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/5 Bee Project/Data Collection/10-17-24/Synced/Bee Hive 1 chunk 1.wav'
basepath = f'/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Acoustic_Py/Investigations/Bees'
filepath = f'{basepath}/{filename}.wav'

y, sr = librosa.load(filepath, sr=None)

# Separate harmonic and percussive components
harmonic, percussive = librosa.effects.hpss(y, margin=(1.0, 1.0))

# Save the harmonic (non-percussive) part
tag_num = 0
sf.write(f'{basepath}/data/{filename}_H{tag_num}.wav', harmonic, sr)
sf.write(f'{basepath}/data/{filename}_P{tag_num}.wav', percussive, sr)