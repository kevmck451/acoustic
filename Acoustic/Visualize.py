# Functions to Visualize Audio

import matplotlib.pyplot as plt
import numpy as np
import Process
import Sample_Library
import Utils


# Function to Display the Waveform
def waveform(Audio_Object):

    # Calculate the time axis in seconds
    time_axis = np.arange(Audio_Object.data.shape[0]) / Audio_Object.SAMPLE_RATE

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot the audio data for the specified channel (assuming channel is 1-indexed)
    ax.plot(time_axis, Audio_Object.data, linewidth=0.5)
    ax.set_ylabel('Amplitude')
    ax.set_ylim([-1, 1])  # set the y-axis limits to -1 and 1
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')  # add horizontal line at y=0
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Visualize: {Audio_Object.filepath.name}')
    fig.tight_layout(pad=1)

    plt.show()

# Function to Display the Bar Graph of Stats
def stats(Audio_Object):

    channel_stats = Audio_Object.stats()

    # Prepare the data for the bar plots
    stat_names = list(channel_stats.keys())
    num_channels = 1
    bar_width = 0.15  # width of each bar
    index = np.arange(num_channels)  # x-axis values
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # colors for each statistic

    # Create the figure and subplots for Max, Min, and Mean
    fig, (ax1, ax4, ax5) = plt.subplots(1, 3, figsize=(14, 4))

    for i, stat_name in enumerate(stat_names[:3]):
        values = [channel_stats[stat_name]]
        ax1.bar(index + i * bar_width, values, bar_width, color=colors[i], label=stat_name)
        ax1.text(index + i * bar_width, values[0], str(round(values[0], 2)), ha='center', va='bottom')
        ax1.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        ax1.set_ylim([-1, 1])
        ax1.set_title('Max, Min, and Mean')
        ax1.set_ylabel('Value')

    # Create subplot for RMS
    values_rms = [channel_stats['RMS']]
    ax4.bar(index, values_rms, bar_width, color=colors[3], label='RMS')
    ax4.text(index, values_rms[0], str(round(values_rms[0], 2)), ha='center', va='bottom')
    ax4.set_ylim([0, 1])
    ax4.set_title('RMS')
    ax4.set_ylabel('Value')

    # Create subplot for Dynamic Range
    values_dyn_range = [channel_stats['Dynamic Range']]
    ax5.bar(index, values_dyn_range, bar_width, color=colors[4], label='Dynamic Range')
    ax5.text(index, values_dyn_range[0], str(round(values_dyn_range[0], 2)), ha='center', va='bottom')
    ax5.set_ylim([0, 2])
    ax5.set_title('Dynamic Range')
    ax5.set_ylabel('Value')

    # Remove the x-axis labels for the subplots
    ax1.set_xticks([])
    ax4.set_xticks([])
    ax5.set_xticks([])

    # Configure the plot
    fig.suptitle(f'Stats: {Audio_Object.filepath.name}', fontsize=16)
    fig.tight_layout(pad=1)

    plt.show()

# Function to Display the Spectral Plot
def spectral_plot(Audio_Object, normalize=True):

    if normalize:
        data = Process.normalize(Audio_Object).data
    else:
        data = Audio_Object.data

        # Define the desired frequency range
    frequency_range = (0, 2000)

    fig, ax = plt.subplots(figsize=(14, 4))

    spectrum = np.fft.fft(data)  # Apply FFT to the audio data
    magnitude = np.abs(spectrum)

    # Calculate frequency bins and positive frequency mask for each sample
    frequency_bins = np.fft.fftfreq(len(data), d=1 / Audio_Object.SAMPLE_RATE)
    positive_freq_mask = (frequency_bins >= frequency_range[0]) & (frequency_bins <= frequency_range[1])

    channel_spectrums = [magnitude[positive_freq_mask][:len(frequency_bins)]]

    # Average across all channels
    average_spectrum = np.mean(channel_spectrums, axis=0)

    ax.plot(frequency_bins[:len(average_spectrum)], average_spectrum, color='b')

    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('Magnitude', fontweight='bold')
    ax.set_title(f'Spectral Plot: {Audio_Object.filepath.name}')
    ax.grid(True)
    fig.tight_layout(pad=1)
    plt.show()

# Function to Display Waveform, Stats, and Spectral Plot
def overview(Audio_Object, save=False, override=False):
    channel_stats = Audio_Object.stats()

    # Create the figure
    fig = plt.figure(figsize=(14, 8))

    # Waveform plot
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    time_axis = np.arange(Audio_Object.data.shape[0]) / Audio_Object.SAMPLE_RATE
    ax1.plot(time_axis, Audio_Object.data, linewidth=0.5)
    ax1.set_ylabel('Amplitude')
    ax1.set_ylim([-1, 1])
    ax1.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax1.set_xlabel('Time (s)')
    ax1.set_title(f'Waveform: {Audio_Object.filepath.name}')

    # Stat plots (Min, Max, Mean)
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    stat_names = ['Min', 'Max', 'Mean']
    values = [channel_stats[stat_name] for stat_name in stat_names]
    ax2.bar(stat_names, values, color=['green', 'blue', 'red'])
    ax2.set_ylim([-1, 2])
    ax2.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax2.set_title(', '.join(stat_names))
    ax2.set_ylabel('Value')

    # Add numeric labels above the bars
    for i, value in enumerate(values):
        ax2.text(i, value, str(round(value, 2)), ha='center', va='bottom')

    # RMS plot
    values_rms = [channel_stats['RMS']]
    index = np.arange(1)  # x-axis values
    ax3 = plt.subplot2grid((3, 3), (1, 1))
    values = [channel_stats['RMS']]
    ax3.bar(['RMS'], values, color='orange')
    ax3.set_ylim([0, 1])
    ax3.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax3.text(index, values_rms[0], str(round(values_rms[0], 2)), ha='center', va='bottom')
    ax3.set_title('RMS')
    ax3.set_ylabel('Value')

    # Dynamic Range plot
    values_dyn_range = [channel_stats['Dynamic Range']]
    ax4 = plt.subplot2grid((3, 3), (1, 2))
    values = [channel_stats['Dynamic Range']]
    ax4.bar(['Dynamic Range'], values, color='purple')
    ax4.set_ylim([0, 2])
    ax4.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax4.text(index, values_dyn_range[0], str(round(values_dyn_range[0], 2)), ha='center', va='bottom')
    ax4.set_title('Dynamic Range')
    ax4.set_ylabel('Value')

    # Spectral plot
    ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    data = Process.normalize(Audio_Object).data
    frequency_range = (0, 2000)
    spectrum = np.fft.fft(data)
    magnitude = np.abs(spectrum)
    frequency_bins = np.fft.fftfreq(len(data), d=1 / Audio_Object.SAMPLE_RATE)
    positive_freq_mask = (frequency_bins >= frequency_range[0]) & (frequency_bins <= frequency_range[1])
    channel_spectrums = [magnitude[positive_freq_mask][:len(frequency_bins)]]
    average_spectrum = np.mean(channel_spectrums, axis=0)
    ax5.plot(frequency_bins[:len(average_spectrum)], average_spectrum, color='black')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Magnitude')
    ax5.set_title(f'Spectral Plot: {Audio_Object.filepath.name}')
    ax5.grid(True)

    # Adjust layout and show the plots
    fig.tight_layout()
    if save:
        save_directory = Sample_Library.VISUALIZE_SAVE_DIRECTORY
        filename = f'{Audio_Object.filepath.stem}.pdf'
        save_as = f'{save_directory}/{filename}'
        Utils.create_directory_if_not_exists(save_directory)

        if Utils.check_file_exists(save_as):
            if override:
                plt.savefig(save_as)
                plt.close(fig)
                print('file saved')
            else:
                plt.close(fig)
        else:
            plt.savefig(save_as)
            plt.close(fig)
            print('file saved')
    else:
        plt.show()

