import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm


# Select and load subject EEG file
eeg_file = "C:\Omar\College\Semester_8\Automata & Computability\Project\ds003004-download\sub-35\eeg\sub-35_task-ImaginedEmotion_eeg.set"
raw = mne.io.read_raw_eeglab(eeg_file, preload=True)

# Drop non-EEG channels. (Artifact monitoring, etc)
extra_channels = [ch for ch in raw.info['ch_names'] if ch.startswith('EXG')]
raw.drop_channels(extra_channels)

channels_info = pd.read_csv("C:\Omar\College\Semester_8\Automata & Computability\Project\ds003004-download\sub-35\eeg\sub-35_task-ImaginedEmotion_channels.tsv", sep = '\t')

electrodes_info = pd.read_csv("C:\Omar\College\Semester_8\Automata & Computability\Project\ds003004-download\sub-35\eeg\sub-35_task-ImaginedEmotion_electrodes.tsv", sep = '\t')

# Aply ICA and (manually) remove artifact components for this subject 
ica = mne.preprocessing.ICA(n_components=30, random_state=42)
ica.fit(raw)
raw = ica.apply(raw)
# Manual selection by visual inspection
artifacts = [0, 1, 2, 3, 4, 9, 11, 13, 14, 15, 18, 19, 20, 23, 24, 26, 28, 29]
ica.exclude = artifacts


# Map events to epochs
events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id = event_id, tmin=-0.2, tmax=1.5 , preload=True)

annotations = pd.DataFrame({
    'onset': raw.annotations.onset,
    
    'duration': raw.annotations.duration,

    'description': raw.annotations.description
})

# epochs.event_id is a dict mapping label (str) -> int.
reverse_ev_id = {v: k for k, v in epochs.event_id.items()}

epoch_times = [
    epochs.events[i, 0] / raw.info['sfreq']
    for i in range(len(epochs.events))
    if epochs.events[i, 2] in reverse_ev_id  # Only include events that have a mapping
]

epoch_labels = [
    reverse_ev_id[event]
    for event in epochs.events[:, 2]
    if event in reverse_ev_id  # Use the reverse mapping to convert integer code to label
]

epochs_df = pd.DataFrame({
    'epoch_start_time': epoch_times,
    'event_label': epoch_labels
})

print(epochs_df.head())

# Plot epochs with their associated event labels
plt.figure(figsize=(10, 6))

# Scatter plot: epochs vs event type
plt.scatter(epochs_df['epoch_start_time'], epochs_df['event_label'], color='red', marker="o")

plt.xlabel("Epoch Start Time (seconds)")
plt.ylabel("Event Label")
plt.title("Event Annotations Mapped to Epochs")
plt.grid(True)

plt.show()

def compute_band_power(epoch, sfreq, bands = {'alpha':(8,12), 'beta':(13,30)}):
    
    psd, freqs = mne.time_frequency.psd_array_welch(epoch, sfreq=sfreq, n_fft=256)
    band_power = {}
    for band, (low,high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_power[band] = np.mean(psd[:, idx_band])
    
    return band_power

features = []
sfreq = raw.info['sfreq']

for epoch in epochs.get_data():

    bp = compute_band_power(epoch,sfreq)
    features.append([bp['alpha'], bp['beta']])

features = np.array(features)

n_emotional_states = 16

model = hmm.GaussianHMM(n_components=n_emotional_states, covariance_type="diag", n_iter=200, random_state=42)
model.fit(features)

hidden_states = model.predict(features)
print("Predicted hidden states: ", hidden_states)