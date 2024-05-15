# Mic Mounts and their characteristics

from utils import CSVFile
import ast

class Mount:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'Untitled')
        self.number_of_mics = kwargs.get('number_of_mics', 4)
        self.mount_geometry = kwargs.get('mount_geometry', None)
        self.channel_position = kwargs.get('channel_positions', None)
        self.cover = kwargs.get('cover', None)

        filepath = kwargs.get('filepath', None)

        if filepath is not None:
            info_file = CSVFile(filepath)

            if self.mount_geometry is None:
                self.mount_geometry = info_file.get_value(self.name, 'Mount')

            if self.channel_position is None:
                self.channel_position = ast.literal_eval(info_file.get_value(self.name, 'Position'))
                if self.number_of_mics == 4:
                    self.mic_driver = self.channel_position[0][0]
                    self.mic_passenger = self.channel_position[0][1]
                    self.mic_bs_driver = self.channel_position[1][0]
                    self.mic_bs_passenger = self.channel_position[1][1]

            if self.cover is None:
                self.cover = info_file.get_value(self.name, 'Cover')

        if self.channel_position is not None and self.number_of_mics == 4:
            self.mic_driver = self.channel_position[0][0]
            self.mic_passenger = self.channel_position[0][1]
            self.mic_bs_driver = self.channel_position[1][0]
            self.mic_bs_passenger = self.channel_position[1][1]

    def __str__(self):
        return f'---------Mic Mount Object---------\n' \
               f'Name: {self.name}\n' \
               f'Mics: {self.number_of_mics}\n' \
               f'Geometry: {self.mount_geometry}\n' \
               f'Mic Position: {self.channel_position}\n' \
               f'Mic Driver: {self.mic_driver}\n' \
               f'Mic Passenger: {self.mic_passenger}\n' \
               f'Mic Backseat Driver: {self.mic_bs_driver}\n' \
               f'Mic Backseat Passenger: {self.mic_bs_passenger}\n' \
               f'Cover: {self.cover}'




if __name__ == '__main__':

    mic_position = [[1, 2],
                    [3, 4]]

    mount_1 = Mount(number_of_mics=4, mount_geometry='square', channel_positions=mic_position, name='Exp')