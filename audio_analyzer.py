#!/usr/bin/env python3
"""
Audio Analysis Application using Librosa and Matplotlib with Zoom Functionality
A comprehensive tool for analyzing audio files with interactive zoom capabilities.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import librosa
import librosa.display
import numpy as np
import os
from typing import Optional, Tuple, Any


class AudioAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analysis Tool - Librosa & Matplotlib with Zoom")
        self.root.geometry("1400x900")
        
        # Audio data
        self.audio_data: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None
        self.file_path: Optional[str] = None
        
        # Zoom functionality
        self.zoom_start: Optional[float] = None
        self.zoom_end: Optional[float] = None
        self.span_selector: Optional[SpanSelector] = None
        self.zoom_enabled = tk.BooleanVar(value=True)
        
        # Analysis results cache
        self.analysis_cache = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Audio File", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=0, padx=(0, 10))
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Analysis selection frame
        analysis_frame = ttk.LabelFrame(main_frame, text="Analysis Functions", padding="5")
        analysis_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Analysis options
        self.analysis_var = tk.StringVar(value="waveform")
        analysis_options = [
            ("Waveform", "waveform"),
            ("Spectrogram", "spectrogram"),
            ("Mel Spectrogram", "mel_spectrogram"),
            ("MFCC", "mfcc"),
            ("Chroma", "chroma"),
            ("Spectral Centroid", "spectral_centroid"),
            ("Zero Crossing Rate", "zcr"),
            #("Tempo & Beat", "tempo_beat"),
            ("Harmonic-Percussive", "harmonic_percussive"),
            ("Onset Detection", "onset_detection"),
            ("Pitch (F0)", "pitch"),
            ("Spectral Rolloff", "spectral_rolloff"),
            ("Spectral Bandwidth", "spectral_bandwidth"),
            ("RMS Energy", "rms_energy"),
            ("Tonnetz", "tonnetz")
        ]
        
        for i, (text, value) in enumerate(analysis_options):
            ttk.Radiobutton(analysis_frame, text=text, variable=self.analysis_var, 
                          value=value).grid(row=i, column=0, sticky=tk.W, pady=2)
        
        # Control buttons
        button_frame = ttk.Frame(analysis_frame)
        button_frame.grid(row=len(analysis_options), column=0, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="Analyze", 
                  command=self.perform_analysis).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Reset Zoom", 
                  command=self.reset_zoom).grid(row=0, column=1, padx=(5, 0))
        
        # Zoom controls
        zoom_frame = ttk.LabelFrame(analysis_frame, text="Zoom Controls", padding="5")
        zoom_frame.grid(row=len(analysis_options)+1, column=0, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Checkbutton(zoom_frame, text="Enable zoom selection", 
                       variable=self.zoom_enabled, 
                       command=self.toggle_zoom).grid(row=0, column=0, sticky=tk.W)
        
        self.zoom_info_label = ttk.Label(zoom_frame, text="Full audio")
        self.zoom_info_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Plot frame
        plot_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="5")
        plot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Audio Information", padding="5")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=4, wrap=tk.WORD)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        info_frame.columnconfigure(0, weight=1)
        
        self.cb = None # aggiunta colorbar
        
        # Initial plot
        self.show_welcome_message()
        
    def show_welcome_message(self):
        """Show welcome message on the plot"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Load an audio file to begin analysis\n\nClick and drag on the waveform to zoom into specific segments', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=14)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
        
    def browse_file(self):
        """Browse and load audio file"""
        file_types = [
            ("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("FLAC files", "*.flac"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=file_types
        )
        
        if file_path:
            self.load_audio_file(file_path)
            
    def load_audio_file(self, file_path: str):
        """Load audio file using librosa"""
        try:
            self.audio_data, self.sample_rate = librosa.load(file_path, sr=None)
            self.file_path = file_path
            self.analysis_cache.clear()  # Clear cache when new file is loaded
            self.reset_zoom()  # Reset zoom when loading new file
            
            # Update UI
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Loaded: {filename}")
            
            # Update info
            duration = len(self.audio_data) / self.sample_rate
            self.update_info(f"File: {filename}\n"
                           f"Sample Rate: {self.sample_rate} Hz\n"
                           f"Duration: {duration:.2f} seconds\n"
                           f"Samples: {len(self.audio_data)}")
            
            # Show waveform by default
            self.analysis_var.set("waveform")
            self.perform_analysis()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio file:\n{str(e)}")
            
    def update_info(self, text: str):
        """Update the info text widget"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, text)
        
    def get_current_audio_segment(self) -> Tuple[np.ndarray, int, int]:
        """Get the current audio segment based on zoom settings"""
        if self.audio_data is None:
            return None, 0, 0
            
        if self.zoom_start is not None and self.zoom_end is not None:
            start_sample = int(self.zoom_start * self.sample_rate)
            end_sample = int(self.zoom_end * self.sample_rate)
            start_sample = max(0, start_sample)
            end_sample = min(len(self.audio_data), end_sample)
            return self.audio_data[start_sample:end_sample], start_sample, end_sample
        else:
            return self.audio_data, 0, len(self.audio_data)
            
    def get_cache_key(self, analysis_type: str) -> str:
        """Generate cache key including zoom information"""
        zoom_key = f"{self.zoom_start}_{self.zoom_end}" if self.zoom_start is not None else "full"
        return f"{analysis_type}_{zoom_key}"
        
    def perform_analysis(self):
        """Perform the selected analysis"""
        if self.audio_data is None:
            messagebox.showwarning("Warning", "Please load an audio file first.")
            return
            
        analysis_type = self.analysis_var.get()
        cache_key = self.get_cache_key(analysis_type)
        
        try:
            # Check cache first
            if cache_key in self.analysis_cache:
                result = self.analysis_cache[cache_key]
            else:
                result = self.compute_analysis(analysis_type)
                self.analysis_cache[cache_key] = result
            
            self.plot_analysis(analysis_type, result)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
            
    def compute_analysis(self, analysis_type: str) -> Any:
        """Compute the specified analysis on current audio segment"""
        y_segment, start_sample, end_sample = self.get_current_audio_segment()
        if y_segment is None:
            return None
            
        sr = self.sample_rate
        
        if analysis_type == "waveform":
            return {"data": y_segment, "start_sample": start_sample, "end_sample": end_sample}
            
        elif analysis_type == "spectrogram":
            return {"data": librosa.stft(y_segment), "start_sample": start_sample}
            
        elif analysis_type == "mel_spectrogram":
            return {"data": librosa.feature.melspectrogram(y=y_segment, sr=sr), "start_sample": start_sample}
            
        elif analysis_type == "mfcc":
            return {"data": librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=13), "start_sample": start_sample}
            
        elif analysis_type == "chroma":
            return {"data": librosa.feature.chroma_stft(y=y_segment, sr=sr), "start_sample": start_sample}
            
        elif analysis_type == "spectral_centroid":
            return {"data": librosa.feature.spectral_centroid(y=y_segment, sr=sr), "start_sample": start_sample}
            
        elif analysis_type == "zcr":
            return {"data": librosa.feature.zero_crossing_rate(y_segment), "start_sample": start_sample}
            
        elif analysis_type == "tempo_beat":
            tempo, beats = librosa.beat.beat_track(y=y_segment, sr=sr, onset_envelope=None)
            return {"tempo": tempo, "beats": beats, "start_sample": start_sample}
            
        elif analysis_type == "harmonic_percussive":
            y_harmonic, y_percussive = librosa.effects.hpss(y_segment)
            return {"harmonic": y_harmonic, "percussive": y_percussive, "start_sample": start_sample}
            
        elif analysis_type == "onset_detection":
            onset_frames = librosa.onset.onset_detect(y=y_segment, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            return {"data": onset_times, "start_sample": start_sample}
            
        elif analysis_type == "pitch":
            pitches, magnitudes = librosa.piptrack(y=y_segment, sr=sr)
            return {"pitches": pitches, "magnitudes": magnitudes, "start_sample": start_sample}
            
        elif analysis_type == "spectral_rolloff":
            return {"data": librosa.feature.spectral_rolloff(y=y_segment, sr=sr), "start_sample": start_sample}
            
        elif analysis_type == "spectral_bandwidth":
            return {"data": librosa.feature.spectral_bandwidth(y=y_segment, sr=sr), "start_sample": start_sample}
            
        elif analysis_type == "rms_energy":
            return {"data": librosa.feature.rms(y=y_segment), "start_sample": start_sample}
            
        elif analysis_type == "tonnetz":
            return {"data": librosa.feature.tonnetz(y=y_segment, sr=sr), "start_sample": start_sample}
            
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
            
    def plot_analysis(self, analysis_type: str, result: Any):
        """Plot the analysis results"""
        if self.cb:
            self.cb.remove()
            self.cb = None
        
        self.ax.clear()
        
        if result is None:
            return
            
        sr = self.sample_rate
        start_time = result.get("start_sample", 0) / sr if "start_sample" in result else 0
        
        if analysis_type == "waveform":
            y = result["data"]
            time = np.linspace(start_time, start_time + len(y) / sr, len(y))
            self.ax.plot(time, y)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title('Waveform')
            
            # Setup zoom selector for waveform
            if self.zoom_enabled.get() and self.span_selector is None:
                self.setup_zoom_selector()
            
        elif analysis_type == "spectrogram":
            D = librosa.amplitude_to_db(np.abs(result["data"]), ref=np.max)
            img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=self.ax,
                                         x_coords=np.linspace(start_time, start_time + D.shape[1] * 512 / sr, D.shape[1]))
            self.ax.set_title('Spectrogram')
            self.cb = self.fig.colorbar(img, ax=self.ax, format='%+2.0f dB')
            
        elif analysis_type == "mel_spectrogram":
            D = librosa.power_to_db(result["data"], ref=np.max)
            img = librosa.display.specshow(D, y_axis='mel', x_axis='time', sr=sr, ax=self.ax,
                                         x_coords=np.linspace(start_time, start_time + D.shape[1] * 512 / sr, D.shape[1]))
            self.ax.set_title('Mel Spectrogram')
            self.cb = self.fig.colorbar(img, ax=self.ax, format='%+2.0f dB')
            
        elif analysis_type == "mfcc":
            data = result["data"]
            img = librosa.display.specshow(data, x_axis='time', ax=self.ax,
                                         x_coords=np.linspace(start_time, start_time + data.shape[1] * 512 / sr, data.shape[1]))
            self.ax.set_title('MFCC')
            self.ax.set_ylabel('MFCC Coefficients')
            self.cb = self.fig.colorbar(img, ax=self.ax)
            
        elif analysis_type == "chroma":
            data = result["data"]
            img = librosa.display.specshow(data, y_axis='chroma', x_axis='time', ax=self.ax,
                                         x_coords=np.linspace(start_time, start_time + data.shape[1] * 512 / sr, data.shape[1]))
            self.ax.set_title('Chroma Features')
            self.cb = self.fig.colorbar(img, ax=self.ax)
            
        elif analysis_type == "spectral_centroid":
            data = result["data"][0]
            times = librosa.frames_to_time(np.arange(len(data)), sr=sr) + start_time
            self.ax.plot(times, data)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Hz')
            self.ax.set_title('Spectral Centroid')
            
        elif analysis_type == "zcr":
            data = result["data"][0]
            times = librosa.frames_to_time(np.arange(len(data)), sr=sr) + start_time
            self.ax.plot(times, data)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Zero Crossing Rate')
            self.ax.set_title('Zero Crossing Rate')
            
        elif analysis_type == "tempo_beat":
            tempo = result["tempo"]
            beats = result["beats"]
            beat_times = librosa.frames_to_time(beats, sr=sr) + start_time
            
            # Get current segment for plotting
            y_segment, _, _ = self.get_current_audio_segment()
            time = np.linspace(start_time, start_time + len(y_segment) / sr, len(y_segment))
            
            self.ax.plot(time, y_segment, alpha=0.6, label='Waveform')
            self.ax.vlines(beat_times, -1, 1, color='r', alpha=0.8, linestyle='--', label='Beats')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title(f'Tempo: {tempo:.1f} BPM')
            self.ax.legend()
            
        elif analysis_type == "harmonic_percussive":
            y_harmonic = result["harmonic"]
            y_percussive = result["percussive"]
            time = np.linspace(start_time, start_time + len(y_harmonic) / sr, len(y_harmonic))
            
            self.ax.plot(time, y_harmonic, alpha=0.7, label='Harmonic')
            self.ax.plot(time, y_percussive, alpha=0.7, label='Percussive')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title('Harmonic-Percussive Separation')
            self.ax.legend()
            
        elif analysis_type == "onset_detection":
            onset_times = result["data"] + start_time
            
            # Get current segment for plotting
            y_segment, _, _ = self.get_current_audio_segment()
            time = np.linspace(start_time, start_time + len(y_segment) / sr, len(y_segment))
            
            self.ax.plot(time, y_segment, alpha=0.6, label='Waveform')
            self.ax.vlines(onset_times, -1, 1, color='r', alpha=0.8, linestyle='--', label='Onsets')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title(f'Onset Detection ({len(onset_times)} onsets)')
            self.ax.legend()
            
        elif analysis_type == "pitch":
            pitches = result["pitches"]
            magnitudes = result["magnitudes"]
            
            # Extract fundamental frequency
            f0 = []
            times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr) + start_time
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                f0.append(pitch if pitch > 0 else np.nan)
            
            self.ax.plot(times, f0)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Frequency (Hz)')
            self.ax.set_title('Fundamental Frequency (F0)')
            
        elif analysis_type == "spectral_rolloff":
            data = result["data"][0]
            times = librosa.frames_to_time(np.arange(len(data)), sr=sr) + start_time
            self.ax.plot(times, data)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Hz')
            self.ax.set_title('Spectral Rolloff')
            
        elif analysis_type == "spectral_bandwidth":
            data = result["data"][0]
            times = librosa.frames_to_time(np.arange(len(data)), sr=sr) + start_time
            self.ax.plot(times, data)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Hz')
            self.ax.set_title('Spectral Bandwidth')
            
        elif analysis_type == "rms_energy":
            data = result["data"][0]
            times = librosa.frames_to_time(np.arange(len(data)), sr=sr) + start_time
            self.ax.plot(times, data)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('RMS Energy')
            self.ax.set_title('RMS Energy')
            
        elif analysis_type == "tonnetz":
            data = result["data"]
            img = librosa.display.specshow(data, y_axis='tonnetz', x_axis='time', ax=self.ax,
                                         x_coords=np.linspace(start_time, start_time + data.shape[1] * 512 / sr, data.shape[1]))
            self.ax.set_title('Tonnetz (Tonal Centroid Features)')
            self.cb = self.fig.colorbar(img, ax=self.ax)
        
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()
        
    def setup_zoom_selector(self):
        """Setup the zoom selector for waveform"""
        if self.span_selector is not None:
            self.span_selector.disconnect_events()
            
        self.span_selector = SpanSelector(
            self.ax, 
            self.on_zoom_select,
            direction='horizontal',
            useblit=True,
            #rectprops=dict(alpha=0.3, facecolor='red'), # modificato ! RIVEDERE
            interactive=True
        )
        
    def on_zoom_select(self, xmin: float, xmax: float):
        """Handle zoom selection"""
        if xmax - xmin < 0.01:  # Minimum zoom duration
            return
            
        self.zoom_start = xmin
        self.zoom_end = xmax
        
        # Update zoom info
        self.zoom_info_label.config(text=f"Zoom: {xmin:.2f}s - {xmax:.2f}s")
        
        # Clear cache for new zoom level
        self.analysis_cache.clear()
        
        # Re-analyze with new zoom
        self.perform_analysis()
        
    def reset_zoom(self):
        """Reset zoom to full audio"""
        self.zoom_start = None
        self.zoom_end = None
        self.zoom_info_label.config(text="Full audio")
        
        # Clear cache
        self.analysis_cache.clear()
        
        if self.span_selector is not None:
            #self.span_selector.disconnect_events()
            self.span_selector.clear()

        # Re-analyze full audio
        if self.audio_data is not None:
            self.perform_analysis()
            
    def toggle_zoom(self):
        """Toggle zoom functionality"""
        if self.analysis_var.get() == "waveform" and self.audio_data is not None:
            self.perform_analysis()


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = AudioAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()