import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm

# --- Load and preprocess the EEG data ---
# Select and load subject EEG file
eeg_file = r"C:\Omar\College\Semester_8\Automata & Computability\Project\ds003004-download\sub-35\eeg\sub-35_task-ImaginedEmotion_eeg.set"
raw = mne.io.read_raw_eeglab(eeg_file, preload=True)

# Drop non-EEG channels (e.g., EXG channels for artifact monitoring)
extra_channels = [ch for ch in raw.info['ch_names'] if ch.startswith('EXG')]
raw.drop_channels(extra_channels)

# (Optional) Load channel and electrode info for reference
channels_info = pd.read_csv(r"C:\Omar\College\Semester_8\Automata & Computability\Project\ds003004-download\sub-35\eeg\sub-35_task-ImaginedEmotion_channels.tsv", sep='\t')
electrodes_info = pd.read_csv(r"C:\Omar\College\Semester_8\Automata & Computability\Project\ds003004-download\sub-35\eeg\sub-35_task-ImaginedEmotion_electrodes.tsv", sep='\t')

# Apply ICA and (manually) remove artifact components for this subject 
ica = mne.preprocessing.ICA(n_components=30, random_state=42)
ica.fit(raw)
raw = ica.apply(raw)
# Manual selection by visual inspection (the indices of artifact components)
artifacts = [0, 1, 2, 3, 4, 9, 11, 13, 14, 15, 18, 19, 20, 23, 24, 26, 28, 29]
ica.exclude = artifacts

# --- Map events to epochs ---
events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.2, tmax=1.5, preload=True)

annotations = pd.DataFrame({
    'onset': raw.annotations.onset,
    'duration': raw.annotations.duration,
    'description': raw.annotations.description
})

# Create reverse mapping: event code (int) -> label (str)
reverse_ev_id = {v: k for k, v in epochs.event_id.items()}

# Filter epochs to include only those events that have a mapping.
epoch_times = [
    epochs.events[i, 0] / raw.info['sfreq']
    for i in range(len(epochs.events))
    if epochs.events[i, 2] in reverse_ev_id
]
epoch_labels = [
    reverse_ev_id[event]
    for event in epochs.events[:, 2]
    if event in reverse_ev_id
]

epochs_df = pd.DataFrame({
    'epoch_start_time': epoch_times,
    'event_label': epoch_labels
})

print(epochs_df.head())

# Plot epochs with their associated event labels
plt.figure(figsize=(10, 6))
plt.scatter(epochs_df['epoch_start_time'], epochs_df['event_label'], color='red', marker="o")
plt.xlabel("Epoch Start Time (seconds)")
plt.ylabel("Event Label")
plt.title("Event Annotations Mapped to Epochs")
plt.grid(True)
plt.show()

# --- Feature extraction (alpha and beta band power) ---
def compute_band_power(epoch, sfreq, bands={'alpha': (8, 12), 'beta': (13, 30)}):
    psd, freqs = mne.time_frequency.psd_array_welch(epoch, sfreq=sfreq, n_fft=256)
    band_power = {}
    for band, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        band_power[band] = np.mean(psd[:, idx_band])
    return band_power

features = []
sfreq = raw.info['sfreq']

for epoch in epochs.get_data():
    bp = compute_band_power(epoch, sfreq)
    features.append([bp['alpha'], bp['beta']])
features = np.array(features)

# --- Build the Constrained Gaussian HMM ---
n_emotional_states = 16  # 16 states corresponding to: relaxed, awe, frustration, joy, anger, happiness, sadness, love, fear, compassion, jealousy, contentment, grief, relief, excitement, and disgust.
# We assume state 0 is "relaxed" and states 1-15 are the emotional states.

# Build a binary mask (16x16) for allowed transitions.
# For state 0 (relaxed): Allow transitions to any state.
# For any emotion state (1 to 15): Allow only self-transition and transition to relaxed (state 0).
mask = np.zeros((n_emotional_states, n_emotional_states))
mask[0, :] = 1  # Relaxed can transition to any state.
for i in range(1, n_emotional_states):
    mask[i, 0] = 1   # Emotion state i -> relaxed is allowed.
    mask[i, i] = 1   # Self-transition is allowed.

# Create a custom HMM subclass that enforces the transition constraints.
class ConstrainedGaussianHMM(hmm.GaussianHMM):
    def __init__(self, mask, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def _do_mstep(self, stats):
        # Call the standard maximization step.
        super()._do_mstep(stats)
        # Enforce the constraints on the transition matrix:
        # Multiply elementwise by the mask, then renormalize each row.
        self.transmat_ *= self.mask
        row_sums = self.transmat_.sum(axis=1, keepdims=True)
        # Avoid division by zero.
        row_sums[row_sums == 0] = 1e-10
        self.transmat_ = self.transmat_ / row_sums

# Instantiate the constrained HMM model.
model = ConstrainedGaussianHMM(mask=mask,
                               n_components=n_emotional_states,
                               covariance_type="diag",
                               n_iter=200,
                               random_state=42)

# Fit the HMM to the extracted features.
model.fit(features)

# Use the fitted model to predict the hidden states.
hidden_states = model.predict(features)
print("Predicted hidden states: ", hidden_states)