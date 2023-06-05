# Comparing Wind Loudness on foam & dead cat vs foam & dead cat & fleece

from Comparisons import Mount_Compare
from Mic_Mount import Mount

''' Mount Position Comparisons'''

directories = [ '../../../Data/Wind Tunnel/Fleece/Exp 1/4Ch',
                '../../../Data/Wind Tunnel/Fleece/Exp 2/4Ch',
                '../../../Data/Wind Tunnel/Fleece/Exp 3/4Ch',
                '../../../Data/Wind Tunnel/Fleece/Exp 4/4Ch']

# In this Position, Air going top to bottom
mic_position_exp1 = [[2, 4],
                     [1, 3]]
mic_position_exp2 =  [[1, 2],
                      [3, 4]]
mic_position_exp3 = [[2, 4],
                     [1, 3]]
mic_position_exp4 = [[1, 2],
                     [3, 4]]


mount_1 = Mount(4, 'square', mic_position_exp1, 'Exp1')
mount_2 = Mount(4, 'square', mic_position_exp2, 'Exp2')
mount_3 = Mount(4, 'square', mic_position_exp3, 'Exp3')
mount_4 = Mount(4, 'square', mic_position_exp4, 'Exp4')

mount_comparison = Mount_Compare(directories, [mount_1, mount_2, mount_3, mount_4])
mount_comparison.position_comparison_individual()
mount_comparison.position_comparison_average(cutoff=9)



