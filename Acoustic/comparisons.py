# Classes for comparisons

from audio_multich import Audio_MC
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.fft import fft
import numpy as np
import os


class Compare:
    def __init__(self, directory):
        self.directory = directory
        self.sample_list = []
        self.channels = 4
        self.FIG_SIZE_SMALL = (14, 4)

        for filename in os.listdir(directory):
            # Check if the file is a file (not a subfolder)
            self.filepath = os.path.join(directory, filename)
            if os.path.isfile(self.filepath):
                if filename.endswith('.wav'):
                    self.sample_list.append(Audio_MC(self.filepath))
                    # print(self.filepath)

        # print(self.files_list)
        # self.sample_list.sort(key=lambda x: x.filename)
        self.sample_list.sort(key=lambda x: int(os.path.splitext(os.path.basename(x.filename))[0]))

    def RMS(self, title):

        # Prepare data for plotting
        # Sort the sample list by sample names
        # self.sample_list.sort(key=lambda x: x.filename)

        sample_names = [sample.filename for sample in self.sample_list]
        channel_labels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Average']

        # Transpose the RMS values
        rms_values = []

        for i in range(len(channel_labels)):
            rms_values.append([sample.stats()[i]['RMS'] for sample in self.sample_list])

        # Create bar plot
        barWidth = 0.15
        r1 = np.arange(len(sample_names))  # positions of bars on x-axis

        fig, ax = plt.subplots(figsize=self.FIG_SIZE_SMALL)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # Plot bars for each channel
        for i, rms in enumerate(rms_values):
            plt.bar([p + barWidth * i for p in r1], rms, color=colors[i % len(colors)], width=barWidth,
                    edgecolor='grey', label=channel_labels[i])

        # Adding xticks
        plt.xlabel('Samples', fontweight='bold')
        plt.ylabel('RMS', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(sample_names))], sample_names)
        plt.title(f'{title} RMS Comparison')
        plt.legend(loc='upper left')
        plt.ylim((0, 3800))
        plt.show()

        for sample in self.sample_list:
            sample_stats = sample.stats()
            print(f'Sample: {sample.filename}')
            for stats in sample_stats:
                channel = stats['Channel']
                rms = stats['RMS']
                print(f"Channel {channel}: RMS={rms}")
            print()

    def Peak(self, title):
        # Prepare data for plotting
        # Sort the sample list by sample names
        # self.sample_list.sort(key=lambda x: x.filename)

        sample_names = [sample.filename for sample in self.sample_list]
        channel_labels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Average']

        # Transpose the peak level values
        peak_values = []

        for i in range(len(channel_labels)):
            peak_values.append([sample.stats()[i]['Max Value'] for sample in self.sample_list])

        # Create bar plot
        barWidth = 0.15
        r1 = np.arange(len(sample_names))  # positions of bars on x-axis

        fig, ax = plt.subplots(figsize=self.FIG_SIZE_SMALL)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # Plot bars for each channel
        for i, peak in enumerate(peak_values):
            plt.bar([p + barWidth * i for p in r1], peak, color=colors[i % len(colors)], width=barWidth,
                    edgecolor='grey', label=channel_labels[i])

        # Adding xticks
        plt.xlabel('Samples', fontweight='bold')
        plt.ylabel('Peak Level', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(sample_names))], sample_names)
        plt.title(f'{title} Peak Level Comparison')
        plt.legend()
        plt.show()

        for sample in self.sample_list:
            sample_stats = sample.stats()
            print(f'Sample: {sample.filename}')
            for stats in sample_stats:
                channel = stats['Channel']
                peak = stats['Max Value']
                print(f"Channel {channel}: Peak Level={peak}")
            print()

    def Range(self, title):
        # Prepare data for plotting
        # Sort the sample list by sample names
        # self.sample_list.sort(key=lambda x: x.filename)

        sample_names = [sample.filename for sample in self.sample_list]
        channel_labels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Average']

        # Calculate the dynamic range values
        dynamic_range_values = []

        for i in range(len(channel_labels)):
            dynamic_range_values.append([sample.stats()[i]['Dynamic Range'] for sample in self.sample_list])

        # Create bar plot
        barWidth = 0.15
        r1 = np.arange(len(sample_names))  # positions of bars on x-axis

        fig, ax = plt.subplots(figsize=self.FIG_SIZE_SMALL)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        # Plot bars for each channel
        for i, dynamic_range in enumerate(dynamic_range_values):
            plt.bar([p + barWidth * i for p in r1], dynamic_range, color=colors[i % len(colors)], width=barWidth,
                    edgecolor='grey', label=channel_labels[i])

        # Adding xticks
        plt.xlabel('Samples', fontweight='bold')
        plt.ylabel('Dynamic Range', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(sample_names))], sample_names)
        plt.title(f'{title} Dynamic Range Comparison')
        plt.legend()
        plt.show()

        for sample in self.sample_list:
            sample_stats = sample.stats()
            print(f'Sample: {sample.filename}')
            for stats in sample_stats:
                channel = stats['Channel']
                dynamic_range = stats['Dynamic Range']
                print(f"Channel {channel}: Dynamic Range={dynamic_range}")
            print()

    def Spectral(self, title):
        # Prepare data for plotting
        # self.sample_list.sort(key=lambda x: x.filename)

        sample_names = [sample.filename for sample in self.sample_list]

        # Define the desired frequency range
        frequency_range = (0, 2000)

        fig, ax = plt.subplots(figsize=self.FIG_SIZE_SMALL)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        for i, sample in enumerate(self.sample_list):
            channel_spectrums = []
            for channel_data in sample.data:
                spectrum = np.fft.fft(channel_data)  # Apply FFT to the audio data
                magnitude = np.abs(spectrum)

                # Calculate frequency bins and positive frequency mask for each sample
                frequency_bins = np.fft.fftfreq(len(channel_data), d=1 / sample.sample_rate)
                positive_freq_mask = (frequency_bins >= frequency_range[0]) & (frequency_bins <= frequency_range[1])

                channel_spectrums.append(magnitude[positive_freq_mask][:len(frequency_bins)])

            # Average across all channels
            average_spectrum = np.mean(channel_spectrums, axis=0)

            ax.plot(frequency_bins[:len(average_spectrum)], average_spectrum, color=colors[i % len(colors)],
                    label=sample_names[i])

        ax.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax.set_ylabel('Magnitude', fontweight='bold')
        ax.set_title(f'{title} Spectral Analysis')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()


class Mount_Compare:
    def __init__(self, directories, Mount_Object_List):
        self.directory = directories
        self.mount_list = Mount_Object_List
        self.sample_list = []
        self.channels = 4
        self.FIG_SIZE_LARGE = (14, 8)
        self.FIG_SIZE_SMALL = (14, 4)


        for directory in directories:
            exp_list = []
            for filename in os.listdir(directory):
                # Check if the file is a file (not a subfolder)
                self.filepath = os.path.join(directory, filename)
                if os.path.isfile(self.filepath):
                    if filename.endswith('.wav'):
                        exp_list.append(Audio_MC(self.filepath))
                        # print(self.filepath)

            try:
                exp_list.sort(key=lambda x: int(os.path.splitext(os.path.basename(x.filename))[0]))
            except:
                self.sample_list.sort(key=lambda x: x.filename)

            # for exp in exp_list:
            #     print(exp)
            self.sample_list.append(exp_list)
            exp_list = []

        # for list in self.sample_list:
        #     for samp in list:
        #         print(samp)
        #     print('---------------')

    def position_comparison_individual(self):
        '''

        :return:
        '''

        driver_RMS = []
        passenger_RMS = []
        bs_driver_RMS = []
        bs_passenger_RMS = []

        for list, mount in zip(self.sample_list, self.mount_list):
            driver_RMS_temp = []
            passenger_RMS_temp = []
            bs_driver_RMS_temp = []
            bs_passenger_RMS_temp = []

            for samp in list:
                # print(mount)
                # print(samp)
                # print(samp.filename)
                # cutoff = samp.filename.split('.')
                # cutoff = int(cutoff[0])


                driver = mount.channel_position[0][0]
                passenger = mount.channel_position[0][1]
                bs_driver = mount.channel_position[1][0]
                bs_passenger = mount.channel_position[1][1]

                # print(f'D: {driver} | P: {passenger}\nBD: {bs_driver} | BP: {bs_passenger}')
                # print(samp.stats())
                # print(samp.stats()[driver-1].get('RMS'))

                stats = samp.stats()
                driver_RMS_temp.append(stats[driver-1]['RMS'])
                passenger_RMS_temp.append(stats[passenger-1]['RMS'])
                bs_driver_RMS_temp.append(stats[bs_driver-1]['RMS'])
                bs_passenger_RMS_temp.append(stats[bs_passenger-1]['RMS'])

            driver_RMS.append(np.mean(driver_RMS_temp).round(0))
            passenger_RMS.append(np.mean(passenger_RMS_temp).round(0))
            bs_driver_RMS.append(np.mean(bs_driver_RMS_temp).round(0))
            bs_passenger_RMS.append(np.mean(bs_passenger_RMS_temp).round(0))

            driver_RMS_temp = []
            passenger_RMS_temp = []
            bs_driver_RMS_temp = []
            bs_passenger_RMS_temp = []

        print(driver_RMS)
        print(passenger_RMS)
        print(bs_driver_RMS)
        print(bs_passenger_RMS)

        # let's assume that the lengths of driver_RMS, passenger_RMS, bs_driver_RMS, and bs_passenger_RMS are the same
        for i in range(len(driver_RMS)):
            # values for this plot
            averages = [driver_RMS[i], passenger_RMS[i], bs_driver_RMS[i], bs_passenger_RMS[i]]

            # categories (positions)
            categories = ['Driver', 'Passenger', 'BS Driver', 'BS Passenger']

            # define colors for each bar
            colors = ['blue', 'green', 'red', 'orange']

            # plot a new figure
            plt.figure(figsize=self.FIG_SIZE_LARGE)

            # plot the bar chart with custom colors
            plt.bar(categories, averages, color=colors)

            # plot the bar chart with custom colors
            bars = plt.bar(categories, averages, color=colors)

            # axis labels and title
            plt.xlabel('Position')
            plt.ylabel('Average RMS')
            plt.ylim((0, 2200)) # Wind Tunnel: (0, 1200)
            plt.title(f'Comparison of Average RMS by Position for Exp {i + 1}')
            # plt.title(f'Comparison of Average RMS by Position for Orlando')

            # loop over bars and add the value on top
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), ha='center', va='bottom')

            # display the chart
            plt.show()

    def position_comparison_average(self, cutoff=0):
        '''
        idk compare positions or whatever
        first param is the cutoff which blah
        '''

        cutoff_speed = cutoff
        driver_RMS = []
        passenger_RMS = []
        bs_driver_RMS = []
        bs_passenger_RMS = []

        for list, mount in zip(self.sample_list, self.mount_list):
            for samp in list:
                # print(mount)
                # print(samp)
                # print(samp.filename)
                # cutoff = samp.filename.split('.')
                # cutoff = int(cutoff[0])

                if cutoff >= cutoff_speed:
                    driver = mount.channel_position[0][0]
                    passenger = mount.channel_position[0][1]
                    bs_driver = mount.channel_position[1][0]
                    bs_passenger = mount.channel_position[1][1]

                    # print(f'D: {driver} | P: {passenger}\nBD: {bs_driver} | BP: {bs_passenger}')
                    # print(samp.stats())
                    # print(samp.stats()[driver-1].get('RMS'))

                    stats = samp.stats()
                    driver_RMS.append(stats[driver-1]['RMS'])
                    passenger_RMS.append(stats[passenger-1]['RMS'])
                    bs_driver_RMS.append(stats[bs_driver-1]['RMS'])
                    bs_passenger_RMS.append(stats[bs_passenger-1]['RMS'])

        # print(driver_RMS)
        # print(passenger_RMS)
        # print(bs_driver_RMS)
        # print(bs_passenger_RMS)

        driver_average = np.mean(driver_RMS).astype(int)
        passenger_average = np.mean(passenger_RMS).astype(int)
        bs_driver_average = np.mean(bs_driver_RMS).astype(int)
        bs_passenger_average = np.mean(bs_passenger_RMS).astype(int)

        print(f'Driver Average: {driver_average}')
        print(f'Passenger Average: {passenger_average}')
        print(f'BS Driver Average: {bs_driver_average}')
        print(f'BS Passenger Average: {bs_passenger_average}')

        # Average values
        averages = [driver_average, passenger_average, bs_driver_average, bs_passenger_average]

        # Categories (positions)
        categories = ['Driver', 'Passenger', 'BS Driver', 'BS Passenger']

        # Define colors for each bar
        colors = ['blue', 'green', 'red', 'orange']
        plt.figure(figsize=self.FIG_SIZE_LARGE)

        # Plotting the bar chart with custom colors
        bars = plt.bar(categories, averages, color=colors)

        # Axis labels and title
        plt.xlabel('Position')
        plt.ylabel('Average RMS')
        plt.title(f'Comparison of Average RMS by Position: Cuttoff {cutoff_speed}')

        # Adding labels (which are the same as the bar values)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), ha='center', va='bottom')

        # Display the chart
        plt.show()

    def fleece_with_without(self):
        pass



