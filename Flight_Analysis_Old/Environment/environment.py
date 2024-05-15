


from utils import CSVFile


import ast


class Environment:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'Untitled')
        self.temperature = kwargs.get('temp', None)
        self.humidity = kwargs.get('humid', None)
        self.pressure = kwargs.get('press', None)
        self.wind_speed = kwargs.get('wind', None)
        self.location = kwargs.get('location', None)
        self.date = kwargs.get('date', None)
        self.time = kwargs.get('time', None)
        filepath = kwargs.get('filepath', None)

        if filepath is not None:
            info_file = CSVFile(filepath)
            self.temperature = ast.literal_eval(info_file.get_value(self.name, 'Temp'))
            self.humidity = ast.literal_eval(info_file.get_value(self.name, 'Humidity'))
            self.pressure = ast.literal_eval(info_file.get_value(self.name, 'Pressure'))
            self.wind_speed = ast.literal_eval(info_file.get_value(self.name, 'Wind'))
            self.location = info_file.get_value(self.name, 'Location')
            self.date = info_file.get_value(self.name, 'Date')
            self.time = info_file.get_value(self.name, 'Time')

    def __str__(self):
        return f'---------Environment Object---------\n' \
               f'Name: {self.name}\n' \
               f'Temperature: {self.temperature} F\n' \
               f'Humidity: {self.humidity} %\n' \
               f'Pressure: {self.pressure} hPa\n' \
               f'Wind Speed: {self.wind_speed} m/s\n' \
               f'Location: {self.location}\n' \
               f'Date: {self.date}\n' \
               f'Time: {self.time}'






