from datetime import datetime, timedelta
import os

def convert(time, start):
    time_format = "%H:%M:%S"
    dt_time = datetime.strptime(time, time_format)
    dt_start = datetime.strptime(start, time_format)

    # Subtraction
    result = dt_time - dt_start

    # Return adjusted time
    return str(timedelta(seconds=result.total_seconds()))

def main():
    flight_name = input('Enter Flight Name: ')
    takeoff_start_time = input('Enter Start Time: ')
    flight_start_time = convert(input('Enter Flight Start Time: '), takeoff_start_time)
    target = True
    target_times = []

    while target:
        target_time = input('Enter Target Time: ')

        if 'stop' in target_time:
            target = False
        else:
            target_times.append(convert(target_time, takeoff_start_time))

    landing_start_time = convert(input('Enter Landing Time: '), takeoff_start_time)
    end = convert(input('Enter End Time: '), takeoff_start_time)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_directory, f"{flight_name}.txt"), "w") as f:
        f.write(f'Takeoff: 00:00:00\n')
        f.write(f'Flight: {flight_start_time}\n')
        f.write(f'Targets: {", ".join(target_times)}\n')
        f.write(f'Landing: {landing_start_time}\n')
        f.write(f'End: {end}\n')

    print("Times have been written to the file.")


if __name__ == '__main__':
    main()
