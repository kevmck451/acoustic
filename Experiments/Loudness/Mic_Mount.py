# Mic Mounts and their characteristics




class Mount:
    def __init__(self, number_of_mics, mount_geometry, channel_positions, name):
        self.number_of_mics = number_of_mics
        self.mount_geometry = mount_geometry
        self.channel_position = channel_positions
        self.name = name


    def __str__(self):
        return f'Name: {self.name}\nPos: {self.channel_position}'
