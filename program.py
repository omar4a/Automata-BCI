import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import sys
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
import time

import mne
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# --- Redirect sys.stdout and sys.stderr to the GUI log widget ---
class ConsoleRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, message):
        if message.strip() != "":
            self.widget.insert(tk.END, message)
            self.widget.see(tk.END)

    def flush(self):
        pass

# --- Custom logging handler to send logging messages to the text widget ---
class TextHandler(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        msg = self.format(record)
        self.widget.insert(tk.END, msg + "\n")
        self.widget.see(tk.END)

# Use interactive (non-blocking) mode for matplotlib.
plt.ion()

class EEGAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Analysis GUI")
        self.geometry("1000x900")
        
        # Variables for file paths and processing results.
        self.eeg_file = ""
        self.channels_file = ""
        self.electrodes_file = ""
        self.raw = None
        self.ev_df = None
        self.intervals = []              # List of (start_time, end_time)
        self.interval_labels = []        # Annotated labels for each interval
        self.feature_interval_times = [] # Midpoints of intervals
        self.features_scaled = None
        self.hidden_states = None
        self.model = None
        
        self.current_interval_index = 0
        
        # For coordinating ICA exclusion input.
        self.ica_exclusion_event = threading.Event()
        self.ica_exclusion_result = None
        self.ica_for_exclusion = None  # To store the ICA instance
        
        self.create_widgets()
        self.setup_logging()
        # Bind a virtual event to show the ICA exclusion dialog.
        self.bind("<<ShowExclusion>>", self.handle_show_exclusion)
    
    def setup_logging(self):
        # Redirect sys.stdout and sys.stderr.
        self.console_redirector = ConsoleRedirector(self.log_text)
        sys.stdout = self.console_redirector
        sys.stderr = self.console_redirector

        # Also attach a logging handler.
        text_handler = TextHandler(self.log_text)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        text_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(text_handler)
        # Set our own logging level; you can adjust this (e.g. level INFO)
        logger.setLevel(logging.INFO)

        # Set noisy module loggers to WARNING.
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("joblib").setLevel(logging.WARNING)
    
    def create_widgets(self):
        # Top frame: file selection & analysis button.
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        btn = tk.Button(top_frame, text="Load Files and Run Analysis", command=self.load_and_run_analysis)
        btn.pack(side=tk.LEFT)
        
        # Log output area.
        log_frame = tk.Frame(self)
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=5)
        tk.Label(log_frame, text="Log Output:").pack(anchor="w")
        self.log_text = ScrolledText(log_frame, height=10, state=tk.NORMAL)
        self.log_text.pack(fill=tk.X, expand=True)
        
        # Middle frame: automata display (left) & interval plot (right).
        display_frame = tk.Frame(self)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.automata_frame = tk.Frame(display_frame, borderwidth=2, relief=tk.GROOVE)
        self.automata_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.interval_frame = tk.Frame(display_frame, borderwidth=2, relief=tk.GROOVE)
        self.interval_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bottom frame: Next Interval button.
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(side=tk.BOTTOM, pady=10)
        next_btn = tk.Button(bottom_frame, text="Next Interval", command=self.show_next_interval)
        next_btn.pack()
    
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
    
    def thread_safe_log(self, msg):
        self.after(0, lambda: self.log(msg))
    
    def load_and_run_analysis(self):
        # Prompt for file selection with clear dialog titles.
        self.eeg_file = filedialog.askopenfilename(
            title="Select EEG file (.set)",
            filetypes=[("EEG files", "*.set"), ("All files", "*.*")]
        )
        if not self.eeg_file:
            messagebox.showerror("Error", "No EEG file selected!")
            return
        self.log(f"EEG file selected: {self.eeg_file}")
        
        self.channels_file = filedialog.askopenfilename(
            title="Select Channels file (.tsv)",
            filetypes=[("TSV files", "*.tsv"), ("All files", "*.*")]
        )
        if self.channels_file:
            self.log(f"Channels file selected: {self.channels_file}")
        else:
            self.log("No Channels file selected.")
        
        self.electrodes_file = filedialog.askopenfilename(
            title="Select Electrodes file (.tsv)",
            filetypes=[("TSV files", "*.tsv"), ("All files", "*.*")]
        )
        if self.electrodes_file:
            self.log(f"Electrodes file selected: {self.electrodes_file}")
        else:
            self.log("No Electrodes file selected.")
        
        # Start heavy processing in a background thread.
        processing_thread = threading.Thread(target=self.background_worker)
        processing_thread.daemon = True
        processing_thread.start()
    
    def background_worker(self):
        try:
            self.thread_safe_log("Loading EEG data...")
            self.raw = mne.io.read_raw_eeglab(self.eeg_file, preload=True)
            extra_channels = [ch for ch in self.raw.info['ch_names'] if ch.startswith('EXG')]
            self.raw.drop_channels(extra_channels)
            self.thread_safe_log(f"Dropped extra channels: {extra_channels}")
            
            try:
                if self.channels_file:
                    self.channels_info = pd.read_csv(self.channels_file, sep='\t')
                    self.thread_safe_log("Channels info loaded.")
                if self.electrodes_file:
                    self.electrodes_info = pd.read_csv(self.electrodes_file, sep='\t')
                    self.thread_safe_log("Electrodes info loaded.")
            except Exception as e:
                self.thread_safe_log(f"Warning: Could not load channels/electrodes info: {e}")
            
            current_hp = self.raw.info.get('highpass', 0)
            self.thread_safe_log(f"Current highpass cutoff: {current_hp}")
            if current_hp < 1.0:
                self.raw.filter(l_freq=1.0, h_freq=None)
                self.thread_safe_log("Applied a 1 Hz high-pass filter.")
            
            self.thread_safe_log("Running ICA...")
            try:
                ica = mne.preprocessing.ICA(n_components=50, random_state=42, n_jobs=10)
            except TypeError:
                ica = mne.preprocessing.ICA(n_components=50, random_state=42)
            ica.fit(self.raw)
            self.thread_safe_log("ICA computed successfully.")
            
            # Schedule ICA component plotting in the main thread.
            self.after(0, lambda: ica.plot_components())
            self.thread_safe_log("Displaying ICA component figures...")
            time.sleep(0.5)  # Give a moment for figures to appear.
            self.ica_for_exclusion = ica
            # Generate a virtual event to show the exclusion dialog.
            self.event_generate("<<ShowExclusion>>", when="tail")
            self.ica_exclusion_event.wait()  # Wait until user input.
            exclude_list = self.ica_exclusion_result
            self.thread_safe_log(f"Excluding ICA components: {exclude_list}")
            ica.exclude = exclude_list
            self.raw = ica.apply(self.raw)
            self.thread_safe_log("ICA correction applied.")
            
            # --- Extract events and build custom intervals.
            events, event_id = mne.events_from_annotations(self.raw)
            sfreq = self.raw.info['sfreq']
            ev_times = events[:, 0] / sfreq
            reverse_event = {v: k for k, v in event_id.items()}
            ev_labels = [reverse_event[ev[2]] for ev in events]
            ev_df = pd.DataFrame({'time': ev_times, 'label': ev_labels})
            self.thread_safe_log("Events extracted:")
            self.thread_safe_log(ev_df.to_string())
            ev_df = ev_df.iloc[5:].reset_index(drop=True)
            self.ev_df = ev_df
            
            # --- Create a "prebase" interval if it exists.
            prebase_interval = None
            # Use case-insensitive matching to avoid issues with letter case.
            prebase_indices = ev_df[ev_df['label'].str.lower() == "prebase"].index
            if len(prebase_indices) > 0 and prebase_indices[0] < len(ev_df) - 1:
                prebase_idx = prebase_indices[0]
                prebase_interval = (ev_df.loc[prebase_idx, 'time'], ev_df.loc[prebase_idx + 1, 'time'])
                self.thread_safe_log(f"Prebase interval: {prebase_interval}")
            
            # --- Filter for events that are only "press1" or "exit".
            filtered_events = ev_df[ev_df['label'].isin(["press1", "exit"])].reset_index(drop=True)
            self.thread_safe_log("Filtered events (only press1 and exit):")
            self.thread_safe_log(filtered_events.to_string())

            # Create temporary lists that begin with the prebase interval if it exists.
            temp_intervals = []
            temp_labels = []
            if prebase_interval is not None:
                temp_intervals.append(prebase_interval)
                temp_labels.append("prebase")

            # Loop through the filtered events to build the remaining intervals.
            for i in range(len(filtered_events) - 1):
                current_label = filtered_events.loc[i, 'label']
                next_label = filtered_events.loc[i+1, 'label']
                start_time = filtered_events.loc[i, 'time']
                end_time = filtered_events.loc[i+1, 'time']
                if current_label == "press1" and next_label == "exit":
                    idx = ev_df[ev_df['time'] == start_time].index
                    if len(idx) > 0 and idx[0] > 0:
                        emotion = ev_df.loc[idx[0]-1, 'label']
                    else:
                        emotion = "unknown"
                    temp_intervals.append((start_time, end_time))
                    temp_labels.append(emotion)
                elif current_label == "exit" and next_label == "press1":
                    temp_intervals.append((start_time, end_time))
                    temp_labels.append("relaxed")

            # Replace the original lists with the newly built ones.
            self.intervals = temp_intervals
            self.interval_labels = temp_labels

            self.thread_safe_log("Identified emotion intervals (in seconds):")
            self.thread_safe_log(str(self.intervals))
            
            # --- Feature Extraction: Use ThreadPoolExecutor with 10 workers.
            def compute_wavelet_features(epoch, sfreq,
                                         bands={'delta': (1,4),
                                                'theta': (4,8),
                                                'alpha': (8,12),
                                                'beta': (13,30),
                                                'gamma': (30,45)}):
                epoch = epoch[np.newaxis, ...]
                frequencies = np.arange(1,46,1)
                n_cycles = frequencies / 3.0
                power = mne.time_frequency.tfr_array_morlet(epoch, sfreq=sfreq,
                                                            freqs=frequencies, n_cycles=n_cycles,
                                                            output='power')
                avg_power = power.mean(axis=(1,3))[0]
                feats = {}
                for band, (fmin, fmax) in bands.items():
                    idx = np.where((frequencies >= fmin) & (frequencies <= fmax))[0]
                    feats[band] = avg_power[idx].mean()
                return feats
            
            def process_interval(interval):
                start_time, end_time = interval
                epoch_data = self.raw.copy().crop(tmin=start_time, tmax=end_time).get_data()
                band_feats = compute_wavelet_features(epoch_data, sfreq)
                midpoint = (start_time + end_time) / 2.0
                return ([band_feats['delta'], band_feats['theta'],
                         band_feats['alpha'], band_feats['beta'], band_feats['gamma']], midpoint)
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(process_interval, self.intervals))
            
            features_list = [res[0] for res in results]
            self.feature_interval_times = [res[1] for res in results]
            features_matrix = np.array(features_list)
            self.thread_safe_log(f"Feature matrix shape: {features_matrix.shape}")
            
            scaler = StandardScaler()
            self.features_scaled = scaler.fit_transform(features_matrix)
            self.thread_safe_log("Features normalized and scaled.")
            
            n_emotional_states = 3
            self.model = hmm.GaussianHMM(n_components=n_emotional_states, covariance_type="full",
                                         n_iter=400, random_state=37)
            self.model.fit(self.features_scaled)
            self.hidden_states = self.model.predict(self.features_scaled)
            self.thread_safe_log(f"Predicted hidden state indices: {self.hidden_states}")
            
            self.after(0, self.init_interval_display)
            self.after(0, self.init_automata_display)
            self.after(0, lambda: self.show_interval(self.current_interval_index))
            
        except Exception as e:
            self.thread_safe_log(f"Error during background processing: {e}")
    
    def handle_show_exclusion(self, event):
        # This handler runs in the main thread.
        self.show_exclusion_dialog(self.ica_for_exclusion)
    
    def show_exclusion_dialog(self, ica):
        # Create a non-modal Toplevel window for ICA exclusion input.
        top = tk.Toplevel(self)
        top.title("ICA Exclusion")
        tk.Label(top, text="Enter comma-separated integer indices for ICA components to exclude:").pack(padx=10, pady=10)
        entry = tk.Entry(top, width=50)
        entry.pack(padx=10, pady=5)
        def on_ok():
            comp_str = entry.get()
            if not comp_str:
                messagebox.showerror("Error", "No components entered. Exiting.")
                top.destroy()
                return
            try:
                exclude_list = [int(s.strip()) for s in comp_str.split(",")]
                self.log(f"User input for ICA exclusion: {exclude_list}")
            except Exception as e:
                messagebox.showerror("Error", f"Invalid input: {e}")
                return
            self.ica_exclusion_result = exclude_list
            self.ica_exclusion_event.set()
            top.destroy()
        tk.Button(top, text="OK", command=on_ok).pack(pady=10)
        entry.focus_set()
    
    def init_automata_display(self):
        self.automata_fig, self.automata_ax = plt.subplots(figsize=(5, 3))
        self.automata_canvas = FigureCanvasTkAgg(self.automata_fig, master=self.automata_frame)
        self.automata_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.draw_automata(highlight_state=None)
    
    def draw_automata(self, highlight_state=None):
        ax = self.automata_ax
        ax.clear()
        # Layout: state 0 at (3,2); state 1 at (1,2); state 2 at (5,2)
        positions = {0: (3,2), 1: (1,2), 2: (5,2)}
        radius = 0.7
        for state, pos in positions.items():
            circle = plt.Circle(pos, radius, color='black', fill=False, lw=3)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], f"{state}", fontsize=20, fontweight='bold', ha='center', va='center')
        # Remove "Start" label and its arrow.
        # Draw arrows between state 0 and state 1, state 0 and state 2.
        # Compute arrow endpoints so they start at boundaries.
        # Arrow between state 0 and 1:
        vec_0_to_1 = np.array(positions[1]) - np.array(positions[0])
        norm_0_to_1 = vec_0_to_1 / np.linalg.norm(vec_0_to_1)
        arrow_start_0_1 = tuple(np.array(positions[0]) + radius * norm_0_to_1)
        arrow_end_0_1   = tuple(np.array(positions[1]) - radius * norm_0_to_1)
        # Arrow between state 0 and 2:
        vec_0_to_2 = np.array(positions[2]) - np.array(positions[0])
        norm_0_to_2 = vec_0_to_2 / np.linalg.norm(vec_0_to_2)
        arrow_start_0_2 = tuple(np.array(positions[0]) + radius * norm_0_to_2)
        arrow_end_0_2   = tuple(np.array(positions[2]) - radius * norm_0_to_2)
        ax.annotate("", xy=arrow_end_0_1, xytext=arrow_start_0_1,
                    arrowprops=dict(arrowstyle="<->", color="black", lw=2))
        ax.annotate("", xy=arrow_end_0_2, xytext=arrow_start_0_2,
                    arrowprops=dict(arrowstyle="<->", color="black", lw=2))
        ax.set_xlim(0,6)
        ax.set_ylim(0,4)
        ax.axis("off")
        if highlight_state is not None:
            pos = positions[highlight_state]
            highlight_circle = plt.Circle(pos, radius+0.05, color='red', fill=False, lw=4)
            ax.add_patch(highlight_circle)
        self.automata_canvas.draw()
    
    def init_interval_display(self):
        self.interval_fig, self.interval_ax = plt.subplots(figsize=(8, 3))
        self.interval_canvas = FigureCanvasTkAgg(self.interval_fig, master=self.interval_frame)
        self.interval_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_interval(self, index):
        if index < 0 or index >= len(self.intervals):
            return
        start_time, end_time = self.intervals[index]
        annotated_label = self.interval_labels[index]
        predicted_state = self.hidden_states[index]
        
        epoch_data = self.raw.copy().crop(tmin=start_time, tmax=end_time).get_data()
        mean_signal = np.mean(epoch_data, axis=0).flatten()
        times = np.linspace(start_time, end_time, len(mean_signal))
        
        self.interval_ax.clear()
        self.interval_ax.plot(times, mean_signal, label=f"Annotated: {annotated_label}", color="blue")
        self.interval_ax.set_title(f"Interval {index+1} | Predicted State: {predicted_state}")
        # Force legend to the top.
        self.interval_ax.legend(loc="upper center")
        self.interval_canvas.draw()
        
        self.draw_automata(highlight_state=predicted_state)
    
    def show_next_interval(self):
        self.current_interval_index += 1
        if self.current_interval_index >= len(self.intervals):
            messagebox.showinfo("Info", "No more intervals.")
            self.current_interval_index = len(self.intervals) - 1
        self.show_interval(self.current_interval_index)

if __name__ == "__main__":
    app = EEGAnalysisGUI()
    app.mainloop()