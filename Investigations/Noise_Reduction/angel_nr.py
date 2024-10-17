


from Filters.noise_reduction import noise_reduction_filter
from Filters.high_pass import high_pass_filter
from Filters.low_pass import low_pass_filter
from Filters.normalize import normalize
from Filters.audio import Audio



if __name__ == '__main__':

    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Field Tests/5 Angel Boom Mount/data/empty_ch4.wav'

    nr_value = 1.5
    stationary = True
    highPass_cutoff = 100
    lowPass_cutoff = 3000

    audio = Audio(filepath=filepath, num_channels=1)

    # audio.data = high_pass_filter(audio, highPass_cutoff)
    # audio.data = low_pass_filter(audio, lowPass_cutoff)

    audio.data = noise_reduction_filter(audio, noise_audio_object=None, std_threshold=nr_value, stationary=stationary)
    # audio.data = noise_reduction_filter(audio, noise_audio_object=None, std_threshold=nr_value, stationary=stationary)
    # audio.data = noise_reduction_filter(audio, noise_audio_object=None, std_threshold=nr_value, stationary=stationary)

    # Ensure the filtered data shape matches the original
    assert audio.data.shape == audio.data.shape, "Filtered data shape does not match original data shape"

    audio.data = normalize(audio)

    path = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Analysis/angel_prop_iso'
    name = f'{audio.name}_NR_{nr_value}.wav'

    audio.export(filepath=path, name=name)