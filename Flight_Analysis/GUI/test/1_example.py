import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout, QPushButton, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio File Analyzer")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.load_button = QPushButton("Load Audio File", self)
        self.load_button.clicked.connect(self.load_audio_file)
        self.layout.addWidget(self.load_button)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

    def load_audio_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Audio File", "", "Audio Files (*.wav *.mp3)", options=options)
        if file_name:
            self.analyze_audio(file_name)

    def analyze_audio(self, file_name):
        # Placeholder for your analysis code
        # Replace this with your backend processing
        print(f"Analyzing {file_name}")

        # Example: Plotting a dummy graph
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot([0, 1, 2, 3], [10, 1, 20, 3])
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
