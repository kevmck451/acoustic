# Comparing Mic Position RMS for Hex Flights from Orlando

from Comparisons import Mount_Compare
from Mic_Mount import Mount

directories = ['../../../Data/Field Tests/Orlando 23/Samples/Full Flight']
# directories = ['../../../Data/Field Tests/Orlando 23/Samples/Flight']

# In this Position, Air going top to bottom
mic_position_exp1 = [[3, 1],
                     [4, 2]]


mount = Mount(4, 'square', mic_position_exp1, 'Orlando')

mount_comparison = Mount_Compare(directories, [mount])
mount_comparison.position_comparison_individual()
# mount_comparison.position_comparison_average()