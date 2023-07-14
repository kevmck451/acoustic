
from Detection.models.template_train_model import Train_Detect_Model

from pathlib import Path
import threading

if __name__ == '__main__':

    dataset = Path('/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/ML Model Data/dataset')

    # create a thread for function with arguments
    t1 = threading.Thread(target=Train_Detect_Model, args=(dataset, 10, 'spectral', 'basic', False))
    t2 = threading.Thread(target=Train_Detect_Model, args=(dataset, 5, 'spectral', 'basic', False))
    t3 = threading.Thread(target=Train_Detect_Model, args=(dataset, 2, 'spectral', 'basic', False))
    t4 = threading.Thread(target=Train_Detect_Model, args=(dataset, 10, 'filter1', 'basic', False))
    t5 = threading.Thread(target=Train_Detect_Model, args=(dataset, 5, 'filter1', 'basic', False))
    t6 = threading.Thread(target=Train_Detect_Model, args=(dataset, 2, 'filter1', 'basic', False))

    # start the thread
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()

    # wait until the thread finishes
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()


    while True:
        t1 = threading.Thread(target=Train_Detect_Model, args=(dataset, 10, 'spectral', 'basic', True))
        t2 = threading.Thread(target=Train_Detect_Model, args=(dataset, 5, 'spectral', 'basic', True))
        t3 = threading.Thread(target=Train_Detect_Model, args=(dataset, 2, 'spectral', 'basic', True))
        t4 = threading.Thread(target=Train_Detect_Model, args=(dataset, 10, 'filter1', 'basic', True))
        t5 = threading.Thread(target=Train_Detect_Model, args=(dataset, 5, 'filter1', 'basic', True))
        t6 = threading.Thread(target=Train_Detect_Model, args=(dataset, 2, 'filter1', 'basic', True))
        t7 = threading.Thread(target=Train_Detect_Model, args=(dataset, 10, 'spectral', 'deep', True))
        t8 = threading.Thread(target=Train_Detect_Model, args=(dataset, 5, 'spectral', 'deep', True))
        t9 = threading.Thread(target=Train_Detect_Model, args=(dataset, 2, 'spectral', 'deep', True))
        t10 = threading.Thread(target=Train_Detect_Model, args=(dataset, 10, 'filter1', 'deep', True))
        t11 = threading.Thread(target=Train_Detect_Model, args=(dataset, 5, 'filter1', 'deep', True))
        t12 = threading.Thread(target=Train_Detect_Model, args=(dataset, 2, 'filter1', 'deep', True))

        # start the thread
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t6.start()
        t7.start()
        t8.start()
        t9.start()
        t10.start()
        t11.start()
        t12.start()

        # wait until the thread finishes
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        t5.join()
        t6.join()
        t7.join()
        t8.join()
        t9.join()
        t10.join()
        t11.join()
        t12.join()
