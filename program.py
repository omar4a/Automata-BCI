import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

##################################
# --- Load and preprocess EEG data ---
##################################

# Select and load subject EEG file
eeg_file = r"C:\Omar\College\Semester_8\Automata & Computability\Project\ds003004-download\sub-34\eeg\sub-34_task-ImaginedEmotion_eeg.set"
raw = mne.io.read_raw_eeglab(eeg_file, preload=True)

# Drop non-EEG channels (e.g., EXG channels for artifact monitoring)
extra_channels = [ch for ch in raw.info['ch_names'] if ch.startswith('EXG')]
raw.drop_channels(extra_channels)

# (Optional) Load channel and electrode info for reference
channels_info = pd.read_csv(r"C:\Omar\College\Semester_8\Automata & Computability\Project\ds003004-download\sub-34\eeg\sub-34_task-ImaginedEmotion_channels.tsv", sep='\t')
electrodes_info = pd.read_csv(r"C:\Omar\College\Semester_8\Automata & Computability\Project\ds003004-download\sub-34\eeg\sub-34_task-ImaginedEmotion_electrodes.tsv", sep='\t')

# --- Verify and apply a 1 Hz high-pass filter before ICA ---
current_hp = raw.info.get('highpass', 0)
print("Current highpass cutoff:", current_hp)
if current_hp < 1.0:
    raw.filter(l_freq=1.0, h_freq=None)
    print("Applied a 1 Hz high-pass filter.")

##################################
# --- Apply ICA and remove artifact components ---
##################################

ica = mne.preprocessing.ICA(n_components=50, random_state=42)
ica.fit(raw)
# Optionally, view the ICA components before exclusion:
# ica.plot_components()
# plt.show()

# Manual component exclusion (determined by visual inspection)
artifacts = [4, 5, 8, 11, 12, 13, 14, 15, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 33, 35, 40, 41, 42, 44, 45, 46, 47]
ica.exclude = artifacts
raw = ica.apply(raw)  # Apply the ICA correction

##################################
# --- Extract events and build custom intervals ---
##################################

# Extract events from annotations.
events, event_id = mne.events_from_annotations(raw)
sfreq = raw.info['sfreq']
ev_times = events[:, 0] / sfreq  # Convert sample numbers to seconds

# Create a reverse mapping: event code -> label
reverse_event = {v: k for k, v in event_id.items()}
ev_labels = [reverse_event[ev[2]] for ev in events]

# Build a DataFrame with events.
ev_df = pd.DataFrame({'time': ev_times, 'label': ev_labels})
print("All events:")
print(ev_df)

# ----- Step 1: Ignore the first few instruction events -----
# Let's assume the first 5 events are instructions; drop them.
ev_df = ev_df.iloc[5:].reset_index(drop=True)

# ----- Step 2: Build intervals corresponding to emotion periods -----
# Assume valid emotion periods are defined by a "press1" followed by an "exit"
# (or vice versa). These intervals reflect periods when the subject is actively experiencing emotion.
intervals = []
for i in range(len(ev_df) - 1):
    current_label = ev_df.loc[i, 'label']
    next_label = ev_df.loc[i + 1, 'label']
    if (current_label == 'press1' and next_label == 'exit') or (current_label == 'exit' and next_label == 'press1'):
        start_time = ev_df.loc[i, 'time']
        end_time = ev_df.loc[i + 1, 'time']
        intervals.append((start_time, end_time))
print("Identified emotion intervals (in seconds):")
print(intervals)

##################################
# --- Feature extraction: Compute wavelet-based features ---
##################################

def compute_wavelet_features(epoch, sfreq, 
                             bands={'delta': (1, 4), 
                                    'theta': (4, 8), 
                                    'alpha': (8, 12), 
                                    'beta': (13, 30), 
                                    'gamma': (30, 45)}):
    """
    Compute average power in several frequency bands using a Morlet wavelet transform.
    """
    # Add an extra dimension: (n_channels, n_times) --> (1, n_channels, n_times)
    epoch = epoch[np.newaxis, ...]
    
    # Define frequencies from 1 to 45 Hz (1 Hz resolution).
    frequencies = np.arange(1, 46, 1)
    n_cycles = frequencies / 3.0  # adjust as needed
    
    # Compute time-frequency representation using Morlet wavelets.
    power = mne.time_frequency.tfr_array_morlet(epoch, sfreq=sfreq, 
                                                freqs=frequencies, n_cycles=n_cycles, 
                                                output='power')
    # Average over channels and time (resulting in one value per frequency).
    avg_power = power.mean(axis=(1, 3))[0]  # shape: (n_frequencies,)
    
    features = {}
    for band, (fmin, fmax) in bands.items():
        idx = np.where((frequencies >= fmin) & (frequencies <= fmax))[0]
        features[band] = avg_power[idx].mean()
    return features

# Build the feature matrix and record midpoints of the intervals.
features_list = []
feature_interval_times = []
for start_time, end_time in intervals:
    # Crop the raw data to the emotion interval.
    epoch_data = raw.copy().crop(tmin=start_time, tmax=end_time).get_data()
    band_features = compute_wavelet_features(epoch_data, sfreq)
    # Order: delta, theta, alpha, beta, gamma.
    features_list.append([band_features['delta'], band_features['theta'], 
                          band_features['alpha'], band_features['beta'], band_features['gamma']])
    feature_interval_times.append((start_time + end_time) / 2.0)

features_matrix = np.array(features_list)
print("Feature matrix shape:", features_matrix.shape)

##################################
# --- Normalize and Scale the Features ---
##################################

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_matrix)

##################################
# --- Build and Fit the HMM ---
##################################

# Here we use a standard GaussianHMM. Adjust n_components as appropriate.
n_emotional_states = 3  # You could set this based on your experimental hypothesis.
model = hmm.GaussianHMM(n_components=n_emotional_states, covariance_type="full", n_iter=400, random_state=37)

# Fit the HMM to the scaled features.
model.fit(features_scaled)

# Predict hidden states from the scaled features.
hidden_states = model.predict(features_scaled)
print("Predicted hidden state indices:", hidden_states)

# Optionally, plot the hidden state timeline (with midpoints of the intervals)
plt.figure(figsize=(10, 5))
plt.plot(feature_interval_times, hidden_states, marker='o', linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("Predicted Hidden State")
plt.title("Hidden State Timeline during Emotion Periods")
plt.grid(True)
plt.show()