# File to characterize a sound source called "Target"

from sample_library import *
import math
import ast

class Target:

    def __init__(self, target_name):
        """
        Constructs all the necessary attributes for the Target object.
        Parameters:
            target_name (str): Type of the target. Expected values are 'tank' or 'generator'.
        """

        type_dict = {'tank': 88, 'decoy': 78}

        self.target_name = target_name
        self.SPL_at_10m_dB = type_dict.get(target_name)

        # convert decibels to intensity
        self.intensity_at_10m_Wm2 = 10 ** (self.SPL_at_10m_dB / 10) * 1e-12



    def propagation_projection_simple(self, distance_increments):
        """
        Calculates sound intensity projections at various distances.

        Parameters:
            distance_increments (int): The number of projections to calculate. Each projection will be 10m further from the source.

        Returns:
            list: A list of decibel values, one for each projection.
        """

        projections = [self.SPL_at_10m_dB]  # start with the original SPL at 10m
        I0 = 1e-12  # Reference intensity

        for i in range(1, distance_increments):
            distance = (i + 1) * 10  # in meters
            intensity = self.intensity_at_10m_Wm2 / (distance ** 2)  # Inverse square law
            decibel = 10 * math.log10(intensity / I0)  # convert intensity to decibels
            projections.append(int(round(decibel, 0)))

        return projections

    def propagation_projection_environment(self, distance_increments, temperature, humidity, pressure):
        """
        Calculates sound intensity projections at various distances considering environmental factors.

        Parameters:
            distance_increments (int): The number of projections to calculate. Each projection will be 10m further from the source.
            temperature (float): The temperature in Fahrenheit.
            humidity (float): The relative humidity in percent.
            pressure (float): The barometric pressure in hPa.

        Returns:
            list: A list of decibel values, one for each projection.
        """
        temperature_celsius = (temperature - 32) * 5 / 9  # Convert temperature to celsius
        pressure_atm = pressure / 1013.25  # Convert pressure to atm
        speed_of_sound = 331.3 * math.sqrt(1 + temperature_celsius / 273.15)  # in m/s, for dry air

        # Adjust for humidity (assuming a simple linear adjustment, which is not strictly correct)
        speed_of_sound *= 1 + 0.02 * humidity / 100

        # Assume pressure has negligible effect on speed of sound
        projections = [self.SPL_at_10m_dB]  # start with the original SPL at 10m
        I0 = 1e-12  # Reference intensity

        for i in range(1, distance_increments):
            distance = (i + 1) * 10  # in meters
            intensity = self.intensity_at_10m_Wm2 / (distance ** 2) * (
                        speed_of_sound / 331.3) ** 2  # Inverse square law, adjusted for speed of sound
            decibel = 10 * math.log10(intensity / I0)  # convert intensity to decibels
            projections.append(int(round(decibel, 0)))

        return projections

    def calculate_distance(self, threshold_dB):
        """
        Calculate the distance at which the sound level from the target would decrease below a given threshold.

        Parameters:
        threshold_dB (float): The sound level threshold in decibels.

        Returns:
        float: The distance in meters at which the sound level from the target would decrease below the threshold.
        """

        # Set the maximum distance (in 10m increments) to project
        # You may want to adjust this value based on your specific needs
        max_distance_increments = 1000

        # Calculate the sound level at various distances
        projections = self.propagation_projection_simple(max_distance_increments)

        # Find the distance at which the sound level drops below the threshold
        for i in range(len(projections)):
            if projections[i] < threshold_dB:
                # Return the distance (in meters) at which the sound level drops below the threshold
                self.threshold_distance = (i + 1) * 10
                # print(f'{(i + 1) * 10} m')
                break

