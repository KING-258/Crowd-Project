import tkinter as tk
from tkinter import messagebox
import json
import time
import random
from threading import Thread

# Load travel commands data
with open("travel_commands.json", "r") as file:
    travel_data = {k.replace("person", ""): v for k, v in json.load(file).items()}

def find_path():
    person_id = entry.get()
    if person_id in travel_data:
        path_display.config(text=(travel_data[person_id]).upper())
    else:
        messagebox.showerror("Error", "Person not found!")

def animate_radar():
    while True:
        for i in range(360):
            canvas.delete("radar_line")
            angle_rad = i * 3.14159 / 180
            x_end = 300 + 250 * tk.cos(angle_rad)
            y_end = 300 - 250 * tk.sin(angle_rad)
            canvas.create_line(300, 300, x_end, y_end, fill="green", tags="radar_line", width=2)
            time.sleep(0.02)

def blink_dots():
    while True:
        canvas.delete("blinking_dot")
        for _ in range(10):
            x, y = random.randint(50, 550), random.randint(50, 550)
            canvas.create_oval(x-3, y-3, x+3, y+3, fill="green", tags="blinking_dot")
        time.sleep(0.5)

# Create main window
root = tk.Tk()
root.title("Crowd Tracker")
root.attributes('-fullscreen', True)
root.configure(bg='black')

# Radar Canvas
canvas = tk.Canvas(root, width=600, height=600, bg="black", highlightthickness=0)
canvas.place(relx=0.5, rely=0.5, anchor="center")

for i in range(1, 6):
    canvas.create_oval(300 - i*100, 300 - i*100, 300 + i*100, 300 + i*100, outline="darkgreen")

canvas.create_line(0, 300, 600, 300, fill="darkgreen")  # Horizontal line
canvas.create_line(300, 0, 300, 600, fill="darkgreen")  # Vertical line

t1 = Thread(target=animate_radar, daemon=True)
t1.start()

t2 = Thread(target=blink_dots, daemon=True)
t2.start()

# Main Frame
frame = tk.Frame(root, bg="black")
frame.place(relx=0.5, rely=0.5, anchor="center")

# Title Label
title = tk.Label(frame, text="Crowd Tracking System", font=("Arial", 16, "bold"), fg="green", bg="black")
title.pack(pady=10)

# Entry field for person ID
entry_label = tk.Label(frame, text="Enter Person Number:", fg="white", bg="black")
entry_label.pack()
entry = tk.Entry(frame)
entry.pack(pady=5)

# Button to find path
find_button = tk.Button(frame, text="Find Path", command=find_path, bg="green", fg="white")
find_button.pack(pady=10)

# Label to display the path
path_display = tk.Label(frame, text="", fg="yellow", bg="black", wraplength=500, justify="center")
path_display.pack(pady=20)

root.mainloop()
