import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from audio_abstract import Audio_Abstract

# Create the main window
app = ctk.CTk()
app.title('UM Acoustic Analysis')

# Set window size
window_width = 1200
window_height = 800
app.geometry(f"{window_width}x{window_height}")
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
center_x = int(screen_width/2 - window_width/2)
center_y = int(screen_height/2 - window_height/2)
app.geometry(f'+{center_x}+{center_y}')

# Add widgets (label, button, etc.)
label = ctk.CTkLabel(app, text='UM Acoustic Analysis', font=('arial', 36))
label.pack(pady=10)

button = ctk.CTkButton(app, text="Click Me")
button.pack(pady=20)

# Matplotlib plot
filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Full Flights/Angel_2.wav'
audio = Audio_Abstract(filepath=filepath)
fig = audio.waveform(ret=True)

# Embedding the plot in the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=app)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill="both", expand=True)

# Draw the canvas
canvas.draw()

# Run the application
app.mainloop()
