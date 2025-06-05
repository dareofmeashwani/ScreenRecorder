import tkinter as tk
from tkinter import ttk
from PIL import Image
import pystray
import threading
import datetime
import os
import sys
import traceback
import base64
from io import BytesIO
from images import base64image
from screen_recorder import ScreenRecorder
import platform

class TaskbarApp:
    def __init__(self):
        """
        Initializes the TaskbarApp.
        """
        self.window = tk.Tk()
        self.window.title("Taskbar Application")
        self.window.geometry("350x350")
        style = ttk.Style()
        style.configure("Faded.TButton", parent="TButton", background="#D0D0D0", foreground="#A0A0A0")
        system_name = platform.system()
        if system_name == "Windows":
            self.window.wm_attributes('-toolwindow', True)
        elif system_name == "Darwin":
            pass
        elif system_name == "Linux":
            self.window.wm_attributes('-type', 'dock')
        # Set the window icon
        try:
            img_bytes = base64.b64decode(base64image)
            img = Image.open(BytesIO(img_bytes))
            buffer = BytesIO()
            img.save(buffer, format="png")  # You can choose other formats if Tkinter supports them
            buffer.seek(0)
            icon_img = tk.PhotoImage(data=buffer.read())  # Use a .png file
            self.window.iconphoto(True, icon_img)
        except Exception as e:
            print(f"Error setting window icon: {e}")
            # Don't raise an exception here; just print a message

        self.label = ttk.Label(self.window, text="Application is running...")
        self.label.pack(pady=20)

        self.start_button = ttk.Button(self.window, text="Start", command=self._start_handler)
        self.start_button.pack(pady=5)
        self.stop_button = ttk.Button(self.window, text="Stop", command=self._stop_handler)
        self.stop_button.pack(pady=5)
        self.stop_button.state(["disabled"])
        self.quit_button = ttk.Button(self.window, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=5)
        
        input_frame = ttk.Frame(self.window, padding="5 5 5 5")
        input_frame.pack(pady=20)
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=3)
        name_label = ttk.Label(input_frame, text="Path")
        name_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.path_location = ttk.Entry(input_frame, width=30)
        self.path_location.insert(0, os.path.join(os.path.expanduser('~'),"Recordings"))
        self.path_location.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

        try:
            img_bytes = base64.b64decode(base64image)
            self.icon_image = Image.open(BytesIO(img_bytes))
        except FileNotFoundError:
            print("Error: icon.png not found. Using a placeholder.")
            self.icon_image = Image.new('RGBA', (64, 64), color=(0, 0, 0, 0))

        self.icon = None
        self._is_running = False  # Use a private variable
        self._recorder = None     # Use a private variable
        self.stop_event = threading.Event()
        self.thread_exception = None
        self.main_thread_id = threading.get_ident()
        self.is_quitting = False

        self.window.protocol("WM_DELETE_WINDOW", self.hide_to_tray)
        self.window.bind("<Unmap>", self.on_unmap)
        self.window.bind("<Map>", self.on_map)
        self.window.withdraw()

    def toggle_fade_by_color(self, button, state):
        if state:
            button.config(style="Faded.TButton")
        else:
            button.config(style="TButton")

    def on_unmap(self, event):
        """
        Handles the window being unmapped (minimized, maximized, or hidden).
        """
        if self.window.wm_state() == 'iconic':
            # This means the window is minimized to the taskbar.
            # We want to hide it to the tray instead.
            self.hide_to_tray()
        elif self.window.wm_state() == 'normal':
            # This case might be triggered by maximizing to full screen,
            # then restoring to normal from the title bar.
            # The 'show_window' method will ensure it's visible and not in tray.
            self.show_window()
        # For maximizing, the state will typically change to 'zoomed' on Windows,
        # or still 'normal' but filling the screen. The <Map> event usually
        # handles bringing it back to the foreground if it was hidden.

    def on_map(self, event):
        """
        Handles the window being mapped (shown).
        """
        # When the window is mapped, ensure it's visible and not in the tray.
        # This handles maximization from the taskbar icon or other means.
        if self.icon and self.icon.visible:
            self.show_window()

    def _start_handler(self):
        """Handles the start button click, preventing multiple starts."""
        if not self._is_running:
            self.start()

    def _stop_handler(self):
        """Handles the stop button click, preventing multiple stops."""
        if self._is_running:
            self.stop()

    def start(self):
        """Starts the background thread."""
        if self._is_running or self._recorder is not None and self._recorder.is_alive():
            return  # Prevent starting if already running or still stopping
        self.start_button.state(["disabled"])
        self.toggle_fade_by_color(self.start_button, True)
        self._is_running = True
        self.label.config(text="Application starting...")
        print("Start button clicked. Application starting.")
        self.stop_event.clear()
        self.thread_exception = None
        self._recorder = ScreenRecorder()
        file_name = str(datetime.datetime.now()).replace(":", "_").replace(".", "_").replace(" ", "_")+".mp4"
        path = self.path_location.get()
        os.makedirs(path, exist_ok=True)
        self._recorder.start(os.path.join(path, file_name))
        self.stop_button.state(["!disabled"])
        self.toggle_fade_by_color(self.stop_button, False)
        self.label.config(text="Application started...")

    def stop(self):
        """Stops the background thread."""
        if not self._is_running or self._recorder is None or not self._recorder.is_alive():
            return  # Prevent stopping if not running or thread not active
        self.stop_button.state(["disabled"])
        self.toggle_fade_by_color(self.stop_button, True)
        self._is_running = False
        self.label.config(text="Stopping application...")
        self._recorder.stop()
        print("Stop button clicked. Requesting application stop.")
        self.stop_event.set()
        self.window.after(100, self._check_thread_status)
        self.start_button.state(["!disabled"])
        self.toggle_fade_by_color(self.start_button, False)

    def _check_thread_status(self):
        """Checks if the background thread has finished."""
        if self._recorder and self._recorder.is_alive():
            self.window.after(100, self._check_thread_status)
        else:
            self._recorder = None
            self.label.config(text="Application stopped.")
            print("Background task stopped.")
            if self.thread_exception:
                print(f"Exception in background thread: {self.thread_exception}")
                traceback.print_exc()

    def show_window(self):
        """Shows the main window."""
        def _show_window():
            if self.is_quitting:
                return
            self.window.deiconify()
            if self.icon:
                self.icon.visible = False
        if threading.get_ident() != self.main_thread_id:
             self.window.after(0, _show_window)
        else:
             _show_window()

    def hide_to_tray(self):
        """Hides the main window and minimizes to the system tray."""
        def _hide_to_tray():
            if self.is_quitting:
                return
            self.window.withdraw()
            if self.icon is None:
                self.buttons = {
                    "Show" : pystray.MenuItem("Show", self.show_window, default=True),
                    "Start": pystray.MenuItem("Start", self._start_handler, enabled=lambda x: not self._is_running),
                    "Stop": pystray.MenuItem("Stop", self._stop_handler, enabled=lambda x: self._is_running),
                    "Quit": pystray.MenuItem("Quit", self.quit_app)
                }
                self.icon = pystray.Icon(
                    "TaskbarApp",
                    self.icon_image,
                    "Taskbar Application",
                    menu=pystray.Menu(
                        self.buttons["Show"],
                        self.buttons["Start"],
                        self.buttons["Stop"],
                        self.buttons["Quit"]
                    ),
                )
                try:
                    self.icon.run_detached()
                except Exception as e:
                    print(f"Error running the icon: {e}")
                    traceback.print_exc()
            elif self.icon:
                self.icon.visible = True

        if threading.get_ident() != self.main_thread_id:
            self.window.after(0, _hide_to_tray)
        else:
            _hide_to_tray()

    def quit_app(self, icon=None, item=None):
        """Quits the application."""
        if self.is_quitting:
            return
        self.is_quitting = True
        print("Quit requested")
        self.stop()

        def _cleanup():
            if self.icon:
                try:
                    self.icon.stop()
                except Exception as e:
                    print(f"Error stopping icon: {e}")
                    traceback.print_exc()
            self.cleanup_and_destroy()

        if self.icon:
            if threading.get_ident() != self.main_thread_id:
                self.window.after(0, _cleanup)
            else:
                _cleanup()
        else:
            self.window.after(0, _cleanup)

    def cleanup_and_destroy(self):
        """Cleanup."""
        try:
            if self.thread_exception:
                print(f"Exception in background thread: {self.thread_exception}")
                traceback.print_exc()
            self.window.destroy()
        except Exception as e:
            print(f"Error destroying window: {e}")
            traceback.print_exc()
        finally:
            sys.exit(0)

    def run(self):
        """Runs the application."""
        self.hide_to_tray()
        self.window.mainloop()

if __name__ == "__main__":
    app = TaskbarApp()
    app.run()