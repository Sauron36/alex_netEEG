import os
import mne
import numpy as np
import torch
from scipy.signal import butter, filtfilt, resample
from torch.utils.data import Dataset, DataLoader


def bandpass_filter(data, low_freq, high_freq, sfreq):
    nyquist = 0.5 * sfreq
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Clamp frequencies to the valid range (0 < Wn <= 1)
    low = max(low, 0.0001)  # Avoid zero or negative frequencies
    high = min(high, 0.99)  # Clamp to Nyquist frequency

    b, a = butter(N=4, Wn=[low, high], btype='band')
    return filtfilt(b, a, data)


def preprocess_eeg(file_path, sfreq=100, duration=60):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    data, _ = raw[:21, :]

    bands = [(1, 7), (8, 30), (31, 100)]
    band_limited_signals = []
    for low, high in bands:
        filtered_data = bandpass_filter(data, low, high, sfreq)
        band_limited_signals.append(filtered_data)

    # Arrange into 21 x 6000 matrices for each band
    matrices = []
    for band_signal in band_limited_signals:
        band_signal = band_signal[:, :sfreq * duration]
        resized_matrix = resample(band_signal, 227, axis=1)
        resized_matrix = resample(resized_matrix, 227, axis=0)
        matrices.append(resized_matrix)

    # Stack matrices for three bands (3, 227, 227)
    return np.stack(matrices, axis=0)  # Shape: (3, 227, 227)


# Preprocessing step: Preprocess and save EEG data
def preprocess_and_save(root_dir, output_dir, sfreq=100, duration=60):
    os.makedirs(output_dir, exist_ok=True)
    problematic_files = []

    class_to_idx = {"normal": 0, "abnormal": 1}

    for label in class_to_idx:
        folder_path = os.path.join(root_dir, label)
        for file in os.listdir(folder_path):
            if file.endswith(".edf"):
                input_path = os.path.join(folder_path, file)
                output_path = os.path.join(output_dir, file.replace(".edf", ".npy"))
                try:
                    preprocessed_data = preprocess_eeg(input_path, sfreq=sfreq, duration=duration)
                    np.save(output_path, preprocessed_data)
                    print(f"Saved: {output_path}") 
                except Exception as e:
                    print(f"Error processing file {input_path}: {e}")
                    problematic_files.append(input_path)

    if problematic_files:
        print("Problematic files:")
        for file in problematic_files:
            print(file)


class EEGDataset(Dataset):
    def __init__(self, preprocessed_dir, transform=None):
        self.preprocessed_dir = preprocessed_dir
        self.transform = transform
        self.data = []

        for label in ['normal', 'abnormal']:  # Adjust if you have different subfolders
            folder_path = os.path.join(preprocessed_dir, label)
            print(f"Checking folder: {folder_path}")
            for file in os.listdir(folder_path):
                if file.endswith(".npy"):
                    self.data.append((os.path.join(folder_path, file), label))  # Store file path with label
                    print(f"Found file: {file}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        preprocessed_data = np.load(file_path)  # Load preprocessed data

        if self.transform:
            preprocessed_data = self.transform(preprocessed_data)

        return torch.tensor(preprocessed_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Preprocess and save the data
train_root_dir = "/Users/saadimran/Desktop/TUH EEG Corpus subset/edf/train"
test_root_dir = "/Users/saadimran/Desktop/TUH EEG Corpus subset/edf/eval"
preprocessed_train_dir = '/Users/saadimran/Desktop/TUH EEG Corpus subset/preprocessed_train'
preprocessed_test_dir = '/Users/saadimran/Desktop/TUH EEG Corpus subset/preprocessed_test'

preprocess_and_save(train_root_dir, preprocessed_train_dir)
preprocess_and_save(test_root_dir, preprocessed_test_dir)

# Instantiate datasets and DataLoaders
train_dataset = EEGDataset(preprocessed_dir=preprocessed_train_dir)
test_dataset = EEGDataset(preprocessed_dir=preprocessed_test_dir)

print(f"Training dataset length: {len(train_dataset)}")
print(f"Testing dataset length: {len(test_dataset)}")


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Example usage
for batch_data, batch_labels in train_loader:
    print(f"Data batch shape: {batch_data.shape}")  # Expected: (8, 3, 227, 227)
    print(f"Labels batch shape: {batch_labels.shape}")  # Expected: (8,)
    break