import tkinter as tk
from tkinter import messagebox

"Import tkinter is a simple gui package, Add imports for face recognition or other features"

# Function for menu options
def authenticate():
    messagebox.showinfo("Authentication code entered here or link to another gui page", "This feature is not implemented yet!")

def view_logs():
    messagebox.showinfo("Logs *same*", "No logs available!")

def quit_application():
    confirm = messagebox.askyesno("Exit *closes gui page* ", "Are you sure you want to quit?")
    if confirm:
        root.destroy()

# Main GUI Application
def create_main_menu():
    global root
    root = tk.Tk()
    root.title("BioPass")
    root.geometry("400x300")  # Set the size of the window

    # Title Label
    tk.Label(root, text="Main", font=("Helvetica", 20)).pack(pady=20)

    # Menu Buttons
    tk.Button(root, text="Authenticate", font=("Helvetica", 14), command=authenticate).pack(pady=10)
    tk.Button(root, text="View Logs", font=("Helvetica", 14), command=view_logs).pack(pady=10)
    tk.Button(root, text="Quit", font=("Helvetica", 14), command=quit_application).pack(pady=10)

    # Run the application
    root.mainloop()

# Run the main menu
if __name__ == "__main__":
    create_main_menu()
