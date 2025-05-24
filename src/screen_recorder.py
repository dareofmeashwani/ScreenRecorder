import mss
import sounddevice as sd
import numpy as np
import cv2
import scipy.io.wavfile as wavfile
import subprocess
import time
import os
import threading
import platform # Import platform module

class ScreenRecorder:
    """
    A class to record desktop screen and audio with start and stop functionality.
    """
    def __init__(self, fps=20, audio_sample_rate=44100, audio_channels=2,
                 final_output_file="output.mp4"):
        """
        Initializes the ScreenRecorder with specified parameters.

        Args:
            fps (int): Frames per second for video recording.
            audio_sample_rate (int): Audio sample rate in Hz.
            audio_channels (int): Number of audio channels (e.g., 1 for mono, 2 for stereo).
        """
        self.current_dir = os.getcwd()
        self.fps = fps
        self.audio_sample_rate = audio_sample_rate
        self.audio_channels = audio_channels
        self.final_output_file = final_output_file

        self._recording = False
        self._video_writer = []
        self._mic_stream = None
        self._mic_frames = []
        self._threads = []

    def _cleanup_temp_files(self):
        """Removes temporary video and audio files."""
        directory = self.current_dir
        
        try:
            for filename in os.listdir(directory):
                if filename.startswith("temp"):
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                        print(f"Deleted: {filepath}")
        except FileNotFoundError:
            print(f"Error: Directory not found at '{directory}'")
        except Exception as e:
            print(f"An error occurred: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for sounddevice to capture audio data."""
        if status:
            print(f"Audio stream status: {status}", flush=True)
        if self._recording:
            self._mic_frames.append(indata.copy())

    def _record_screen(self, **kwargs):
        """Captures screen frames and writes them to a video file."""
        monitor = kwargs.get("monitor")
        filename = kwargs.get("filename")
        width = monitor["width"]
        height = monitor["height"]
        print(f"Detected screen resolution: {width}x{height}", filename, monitor)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
        self._video_writer.append(writer)
        sct = mss.mss()
        start_time = time.time()
        frame_count = 0

        while self._recording:
            try:
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                frame_count += 1

                elapsed_time = time.time() - start_time
                time_to_wait = (frame_count / self.fps) - elapsed_time
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
            except Exception as e:
                print(f"Error during screen capture: {e}")
                self._recording = False
                break

        print(f"Screen capture thread finished. Total frames: {frame_count}")

    def _get_audio_input_device_index(self):
        """
        Attempts to find a suitable audio input device for system audio loopback
        or microphone, based on the operating system.
        """
        devices = sd.query_devices()
        system_name = platform.system()
        chosen_device_index = None
        print("\nAvailable audio input devices (Name, HostAPI, Max Input Channels):")
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((i, device))
                print(f"  {i}: {device['name']} ({sd.query_hostapis(device['hostapi'])['name']}) (Input Channels: {device['max_input_channels']})")

        print("\nAttempting to select audio input device for recording system audio:")

        if system_name == "Windows":
            # On Windows, look for "Stereo Mix", "What U Hear", or WASAPI loopback devices
            preferred_keywords = ["stereo mix", "what u hear", "loopback", "sum"] # Added 'sum' for some virtual cable setups
            for i, device in input_devices:
                device_name_lower = device['name'].lower()
                if any(keyword in device_name_lower for keyword in preferred_keywords):
                    chosen_device_index = i
                    print(f"  --> Auto-selected Windows system audio device: {device['name']}")
                    break
        elif system_name == "Darwin": # macOS
            # On macOS, common virtual audio devices are "iShowU Audio Capture" or "BlackHole"
            preferred_keywords = ["ishowu audio capture", "blackhole"]
            for i, device in input_devices:
                device_name_lower = device['name'].lower()
                if any(keyword in device_name_lower for keyword in preferred_keywords):
                    chosen_device_index = i
                    print(f"  --> Auto-selected macOS system audio device: {device['name']}. "
                          "Ensure it's configured as a multi-output device in Audio MIDI Setup.")
                    break
        elif system_name == "Linux":
            # On Linux (PulseAudio), monitor sources are used for loopback
            # A common pattern is <output_sink_name>.monitor
            # This requires 'pactl load-module module-loopback' to be run by the user.
            for i, device in input_devices:
                # Prioritize devices explicitly marked as monitors
                if ".monitor" in device['name'].lower() and device['hostapi'] == sd.query_hostapis(0)['index']: # Check if it's a PulseAudio monitor
                    chosen_device_index = i
                    print(f"  --> Auto-selected Linux (PulseAudio) monitor device: {device['name']}. "
                          "Ensure 'module-loopback' is loaded in PulseAudio (e.g., 'pactl load-module module-loopback').")
                    break

        if chosen_device_index is None:
            # Fallback to the default input device if no specific system audio device is found
            # This will usually be the microphone.
            default_input_device_index = sd.default.device[0]
            if default_input_device_index >= 0:
                chosen_device_index = default_input_device_index
                print(f"  --> No specific system audio device found. Using default input device: "
                      f"{devices[chosen_device_index]['name']} (This is likely your microphone).")
            else:
                print("  --> No input audio device found at all. Audio recording will be skipped.")
                return None # Indicate no suitable device

        return chosen_device_index

    def _record_mic_audio(self):
        """Captures audio frames using sounddevice."""
        device_index = self._get_audio_input_device_index()

        if device_index is None:
            print("Skipping audio recording due to no suitable input device.")
            self._recording = False # Ensure recording stops if audio setup fails
            return

        try:
            device_info = sd.query_devices(device_index, kind='input')
            max_input_channels = device_info['max_input_channels']

            actual_channels = min(self.audio_channels, max_input_channels)
            if actual_channels < self.audio_channels:
                print(f"Warning: Requested {self.audio_channels} audio channels, but device '{device_info['name']}' only supports {max_input_channels}. Using {actual_channels} channels.")
            else:
                print(f"Using {actual_channels} audio channels from device: {device_info['name']}.")

            self._mic_stream = sd.InputStream(
                samplerate=self.audio_sample_rate,
                channels=actual_channels,
                callback=self._audio_callback,
                device=device_index
            )
            with self._mic_stream:
                while self._recording:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error during audio capture: {e}")
            self._recording = False
        print("Audio capture thread finished.")

    def start(self, file_name= None):
        """
        Starts the screen and audio recording.
        """
        if self._recording:
            print("Recording is already in progress.")
            return
        self._cleanup_temp_files()
        print("Starting screen and audio recording...")
        self._recording = True
        if file_name:
            self.final_output_file = file_name
        self._mic_frames = []

        sct = mss.mss()
        # sct.monitors[0] refers to all monitors combined.
        # sct.monitors[1] is typically the primary monitor on most systems.
        # This might need adjustment if the user has a specific monitor setup.
        monitors = []
        if len(sct.monitors) > 1:
            monitors = sct.monitors[1:] # Assuming primary monitor is sct.monitors[1]
        else:
            monitors = [sct.monitors[0]] # Fallback if only one monitor is detected
        for index in range(len(monitors)):
            self._threads.append(threading.Thread(target=self._record_screen,
                kwargs={"monitor":monitors[index], "filename": os.path.join(self.current_dir, "temp_video_{0}.avi".format(index))}))

        self._threads.append(threading.Thread(target=self._record_mic_audio))
        for thread in self._threads:
            thread.start()
        print("Recording started. Call .stop() to end the recording.")

    def stop(self):
        """
        Stops the screen and audio recording, saves temporary files,
        and merges them into a final video file using ffmpeg.
        """
        if not self._recording:
            print("No recording is currently active.")
            return

        print("Stopping recording...")
        self._recording = False

        for thread in self._threads:
            thread.is_alive() and thread.join()

        for v_writer in self._video_writer:
            v_writer.release()
        
        temp_mic_audio_file = os.path.join(self.current_dir,"temp_mic_audio.wav")
        
        if self._mic_frames:
            recorded_audio = np.concatenate(self._mic_frames, axis=0)
            wavfile.write(temp_mic_audio_file, self.audio_sample_rate, recorded_audio)
            print(f"Audio saved to {temp_mic_audio_file}")
        else:
            print("No audio recorded or audio recording failed.")

        # --- Merge Video and Audio using FFmpeg ---
        filename, file_extension = os.path.splitext(self.final_output_file)
        for v_index in range(len(self._video_writer)):
            temp_video_file = os.path.join(self.current_dir,"temp_video_{0}.avi".format(v_index))
            if os.path.exists(temp_video_file) and os.path.exists(temp_mic_audio_file):
                outfile = filename+"_{0}".format(v_index) + file_extension
                print(f"Merging video ({temp_video_file}) and audio ({temp_mic_audio_file}) into {outfile} using FFmpeg...")
                try:
                    ffmpeg_command = [
                        'ffmpeg',
                        '-i', temp_mic_audio_file,
                        '-i', temp_video_file,
                        '-c:v', 'libx264',
                        '-preset', 'fast',
                        '-crf', '23',
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-shortest',
                        '-y', # Overwrite output file without asking
                        outfile
                    ]
                    subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                    print(f"Successfully created {outfile}")
                except FileNotFoundError:
                    print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH.")
                    print("You can download FFmpeg from: https://ffmpeg.org/download.html")
                except subprocess.CalledProcessError as e:
                    print(f"Error during FFmpeg merging:")
                    print(f"Command: {' '.join(e.cmd)}")
                    print(f"Return Code: {e.returncode}")
                    print(f"STDOUT: {e.stdout}")
                    print(f"STDERR: {e.stderr}")
            elif os.path.exists(temp_video_file) and not os.path.exists(temp_mic_audio_file):
                print(f"Only video recorded. Saving video as {self.final_output_file} (no audio).")
                try:
                    # If no audio was recorded, just rename the video file
                    os.rename(temp_video_file, self.final_output_file)
                    print(f"Successfully saved {self.final_output_file} (video only).")
                except OSError as e:
                    print(f"Error renaming video file: {e}")
            else:
                print("Cannot merge: One or both temporary video/audio files are missing.")
        
        self._cleanup_temp_files()

        print("Recording stopped.")
        self._threads = []
        self._video_writer = []
        self._mic_stream = None

    def is_alive(self):
        for thread in self._threads:
            if thread.is_alive():
                return True
        return False
