# Comparing Wind Loudness on foam & dead cat vs foam & dead cat & fleece

from comparisons import Compare
from comparisons import Mount_Compare
from mic_mount import Mount

''' RMS Comparison'''
directory = '../../../Data/Wind Tunnel/Fleece/Exp 1/4Ch'
Fleece_1 = Compare(directory)
Fleece_1.RMS('Fleece Exp 1')

directory = '../../../Data/Wind Tunnel/Fleece/Exp 2/4Ch'
Fleece_2 = Compare(directory)
Fleece_2.RMS('Fleece Exp 2')

directory = '../../../Data/Wind Tunnel/Fleece/Exp 3/4Ch'
Fleece_3 = Compare(directory)
Fleece_3.RMS('Fleece Exp 3')

directory = '../../../Data/Wind Tunnel/Fleece/Exp 4/4Ch'
Fleece_4 = Compare(directory)
Fleece_4.RMS('Fleece Exp 4')

