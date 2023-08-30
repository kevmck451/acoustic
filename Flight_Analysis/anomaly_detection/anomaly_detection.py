

from Acoustic.audio_abstract import Audio_Abstract
import Acoustic.process as process

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import time


class Anomaly_Detection:

    def __init__(self, audio_object):
        data = process.spectrogram(audio_object, range=(50, 11000))
        self.data = np.transpose(data, (1, 2, 0))
        print(self.data)
        self.img_x = int(self.data.shape[2])
        self.img_y = int(self.data.shape[1])
        self.img_bands = int(self.data.shape[0])

    # Function to create a material from single pixel in image
    def graph_spectra(self, location, title, single):
        values_single = []
        for i in range(self.img_bands):
            values_single.append(self.data[location[1], location[0], i])

        x_a = np.linspace(0, self.img_bands, num=self.img_bands)
        plt.plot(x_a, values_single, linewidth=2, label=title)

        plt.xlabel('time')
        plt.ylabel('frequency')
        # plt.legend(loc='upper right')
        if single:
            plt.show()

    # Function to graph all spectral signature for every pixel in image
    def graph_spectra_all(self):

        x_list = np.linspace(0, (self.img_x - 1), 1)
        y_list = np.linspace(0, (self.img_y - 1), 1)

        for i in range(0, self.img_x - 1):
            for j in range(0, self.img_y - 1):
                self.graph_spectra([int(i), int(j)], 'Full', False)
        plt.show()

    # Function to categorize each pixel based on spectral similarity
    def categorize_pixels_1(self, band='M', variation=32, iterations=19, av=False):
        print('Categorizing')
        stats = False
        self.subcat_variation = variation
        if band == 'S':
            bands = [4, 10, 18, 24, 34, 42, 52, 66]  # reduced bands
        elif band == 'M':
            bands = [2, 6, 10, 14, 18, 22, 34, 42, 52, 56, 66]  # reduced bands
        elif band == 'aviris':
            bands = [14, 23, 32, 47, 69, 98, 131, 184]  # aviris images
        elif band == 'mat':
            bands = [10, 18, 26, 40, 55, 70, 95, 124, 160]  # .mat images
        elif band == 'bot':
            bands = [13, 20, 36, 41, 52, 70, 90, 121]  # bot image
        elif band == 'pika':
            bands = [0,2,4,8,16,32,64,128,256]  # pika image
        else:
            bands = band  # reduced bands

        if stats: print(f'Bands: {bands}')
        # ----------------------

        cll = (variation / 100) - 1
        clh = (variation / 100) + 1
        x_list = [i for i in range(self.img_x)]
        y_list = [i for i in range(self.img_y)]
        # print(self.img_x, self.img_y)

        # if there are bands outside the max num bands, remove
        while np.max(bands) >= self.img_bands: bands.pop()

        # INITIATE ALL PIXEL OBJECTS AND SAMPLE VALUES AT POINTS
        # print('Initializing Pixel Objects')
        t3 = time.time()
        pixel_values = [[[self.data[j, i, k] for k in bands] for j in y_list] for i in x_list]
        self.pixel_master_list = []
        for i in x_list:
            for j in y_list:
                p = pixel_class([i, j], np.asarray(pixel_values[i][j]))
                self.pixel_master_list.append(p)

        # print(f'Pixel Ob Time: {(time.time() - t3) / 60} mins')
        # print(f'Pix / s: {(len(x_list)*len(y_list) / (time.time() - t3))}')

        # CATEGORIZE ALL PIXELS BASED ON SIMILARITY
        def is_similar(compare_list, list):
            similar = [1] * len(compare_list)
            comp = []
            for x, y in zip(compare_list, list):
                if (x * cll) <= y <= (x * clh):
                    comp.append(1)
                else:
                    comp.append(0)
            if comp == similar:
                return True
            else:
                return False

        def check_uncat():
            # print('Checking if any Uncategorized')
            for pixel in self.pixel_master_list:
                if pixel.subcat_num == 0:
                    # print('Found a 0')
                    return True
            # print('Everything is Categorized')
            return False

        # Categorizing ------------------------------------------

        self.subcat_master_list = []
        subcat_num = 1
        c = category_class()
        c.subcat_num = subcat_num
        self.subcat_master_list.append(c)
        f = True
        comp = []
        hh, progress = 0, int((len(self.pixel_master_list)) / 4)

        if stats: print('--First Round Categorizing--')
        for z, pixel in enumerate(self.pixel_master_list):
            if f:
                comp = pixel.values
                pixel.subcat_num = subcat_num
                c.pixel_list.append(pixel)
                pixel.subcategory = c
                f = False
                continue
            if is_similar(comp, pixel.values):
                # print('Similar')
                pixel.subcat_num = subcat_num
                c.pixel_list.append(pixel)
                pixel.subcategory = c

            if stats and z % progress == 0:
                hh += 25
                print(f'{hh}% Categorized')

        category_class.max_subcategories = 1
        # print('Subcat Num: {}'.format(subcat_num))

        if stats: print(f'100% Categorized')

        # ii, prog = 0, (s / 4)
        # While Loop ------------------
        while check_uncat():

            subcat_num += 1
            c = category_class()
            c.subcat_num = subcat_num
            self.subcat_master_list.append(c)
            comp_w = []
            for pixel in self.pixel_master_list:
                if pixel.subcat_num == 0:
                    comp_w = pixel.values
                    pixel.subcat_num = subcat_num
                    c.pixel_list.append(pixel)
                    pixel.subcategory = c
                    break
            for pixel in self.pixel_master_list:
                if pixel.subcat_num == 0 and is_similar(comp_w, pixel.values):
                    pixel.subcat_num = subcat_num
                    c.pixel_list.append(pixel)
                    pixel.subcategory = c
            if subcat_num > iterations:
                for pixel in self.pixel_master_list:
                    if pixel.subcat_num == 0:
                        pixel.subcat_num = subcat_num
                        c.pixel_list.append(pixel)
                        pixel.subcategory = c
                break
            category_class.max_subcategories += 1
            # print('Subcat Num: {}'.format(subcat_num))
        # End of While Loop ---------------
        # print(f'Total Categories: {subcat_num}')
        # Everything is Categorized

        self.subcategory_data_dict = None

        def cat_tally_sort():

            subcat_list = []
            for pixel in self.pixel_master_list:
                subcat_list.append(pixel.subcat_num)
            # Create dict of category numbers
            subcat_dict = {}
            for i in range(1, subcat_num + 1):
                # print(i)
                subcat_dict.update({i: subcat_list.count(i)})
            # print(cat_dict)
            # Sort dict in reverse order based on values
            subcat_dict = dict(sorted(subcat_dict.items(), key=lambda item: item[1], reverse=True))
            # print(cat_dict)

            new_subcat_num_dict = {}
            for i, x in zip(range(1, len(subcat_dict) + 1), subcat_dict.keys()):
                new_subcat_num_dict.update({x: i})

            # print(new_cat_num_dict)

            for pixel in self.pixel_master_list:
                # print(pixel.cat_num, new_cat_num_dict.get(pixel.category))
                if pixel.subcat_num != new_subcat_num_dict.get(pixel.subcat_num):
                    pixel.subcat_num = new_subcat_num_dict.get(pixel.subcat_num)

            subcat_list = []
            for pixel in self.pixel_master_list:
                subcat_list.append(pixel.subcat_num)

            # Create dict of category numbers
            subcat_dict = {}
            for i in range(1, subcat_num + 1):
                subcat_dict.update({i: subcat_list.count(i)})

            # Sort dict in reverse order based on values
            self.subcategory_data_dict = dict(sorted(subcat_dict.items(), key=lambda item: item[1], reverse=True))
            print(self.subcategory_data_dict)

            for subcat in self.subcat_master_list:
                if subcat.subcat_num != new_subcat_num_dict.get(subcat.subcat_num):
                    subcat.subcat_num = new_subcat_num_dict.get(subcat.subcat_num)

        if stats: print('Sorting')
        cat_tally_sort()

        cutoff_percent = 100
        cutoff = (len(self.subcategory_data_dict) + 1) * (cutoff_percent / 100)
        cutoff = int((len(self.subcategory_data_dict) + 1) - cutoff)
        # print(cutoff)
        self.category_matrix = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=float)
        for pixel in self.pixel_master_list:
            if pixel.subcat_num > cutoff:
                self.category_matrix[pixel.location[1], pixel.location[0]] = pixel.subcat_num
            else:
                self.category_matrix[pixel.location[1], pixel.location[0]] = 0

        if av:
            print('Averaging Subcategories')
            average = []
            first = True

            for item in self.subcategory_data_dict.items():
                # print(f'Averaging Subcategory {item[0]}')
                for subcat in self.subcat_master_list:
                    if subcat.subcat_num == item[0]:
                        for pix in subcat.pixel_list:
                            values_single = [self.data[pix.location[1], pix.location[0], i] for i in
                                             range(self.img_bands)]
                            if item[1] == 1:
                                average = values_single
                                continue
                            else:
                                if first:
                                    average = values_single
                                new_list = [((a + b) / 2) for a, b in zip(average, values_single)]
                                average = new_list
                        subcat.average_values = average
                        average = []


# PIXEL CLASS TO STORE LOCATION, VALUES, AND CATEGORY
class pixel_class:
    def __init__(self, location, values):
        self.location = location
        self.values = values
        self.cat_num = 0
        self.subcat_num = 0
        self.subcategory = object

# CATEGORY CLASS TO STORE CAT, SUBCAT, CAT TYPE, SUBCAT TYPE, CAT AV VALUE
class category_class:
    category_list = ['unknown', 'natural', 'manmade', 'noise']
    natural_sub_list = ['unknown', 'vegetation', 'water', 'soil', 'rock']
    manmade_sub_list = ['unknown', 'metal', 'plastic', 'path', 'wood', 'concrete']

    max_subcategories = 0

# TIME CLASS TO GIVE STATS ABOUT HOW LONG FUNCTION TAKES
class time_class:
    def __init__(self, name):
        self.start_time = time.time()
        self.name = name

    def stats(self):
        total_time = round((time.time() - self.start_time), 1)

        if total_time < 60:
            print(f'{self.name} Time: {total_time} secs')
        else:
            total_time_min = round((total_time / 60), 1)
            print(f'{self.name} Time: {total_time_min} mins')

        return total_time


if __name__ == '__main__':

    filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Engines/Spring Samples/NM_D_X.wav'
    audio = Audio_Abstract(filepath=filepath)

    anomaly_detector = Anomaly_Detection(audio)
    anomaly_detector.graph_spectra_all()

