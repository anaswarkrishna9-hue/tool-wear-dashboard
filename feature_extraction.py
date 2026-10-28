import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

def extract_features_from_mill(mat_file_path='mill.mat'):
    """
    Extracts time-domain and frequency-domain features from the NASA milling dataset.
    Only runs with verified, non-null flank wear measurements (VB) are kept.
    No synthetic or invented labels are created.
    """
    print(f"Loading '{mat_file_path}'...")
    mat = sio.loadmat(mat_file_path)
    data = mat['mill'][0]
    
    sensor_channels = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']
    sampling_rate = 250.0  # Hz
    
    rows = []
    dropped_runs = []
    
    for idx, sample in enumerate(data):
        case_id = int(sample['case'].squeeze())
        cut_id = int(sample['run'].squeeze())
        run_number = idx + 1
        vb_raw = sample['VB']
        
        # 1. Check label validity (do not assume or invent missing labels)
        if vb_raw.size == 0:
            dropped_runs.append((run_number, case_id, cut_id, "Empty array"))
            continue
            
        vb_val = float(vb_raw.squeeze())
        if np.isnan(vb_val):
            dropped_runs.append((run_number, case_id, cut_id, "NaN (unmeasured wear)"))
            continue

        # 2. Check sensor signal integrity (filter hardware/DAQ corruption spikes > 15V)
        # Physical Matsuura MC-510V DAQ range is +/- 10V
        corrupted = False
        for ch in sensor_channels:
            sig_test = sample[ch].flatten()
            if np.any(np.isnan(sig_test)) or np.any(np.isinf(sig_test)) or np.max(np.abs(sig_test)) > 15.0:
                dropped_runs.append((run_number, case_id, cut_id, f"Corrupted DAQ signal in {ch} (>15V)"))
                corrupted = True
                break
        if corrupted:
            continue
            
        # Cutting parameters
        doc = float(sample['DOC'].squeeze())
        feed = float(sample['feed'].squeeze())
        material = int(sample['material'].squeeze())  # 1 = Cast iron, 2 = Steel
        
        row_dict = {
            'run_id': idx + 1,
            'case': case_id,
            'cut': cut_id,
            'DOC': doc,
            'feed': feed,
            'material': material,  # 1: Cast Iron, 2: Steel
            'VB': vb_val
        }
        
        # Define Tool Wear Condition Class for classification
        if vb_val < 0.20:
            row_dict['wear_class'] = 'Healthy'
            row_dict['wear_code'] = 0
        elif vb_val <= 0.40:
            row_dict['wear_class'] = 'Warning'
            row_dict['wear_code'] = 1
        else:
            row_dict['wear_class'] = 'Critical'
            row_dict['wear_code'] = 2
            
        # Extract features for each of the 6 sensor channels
        for ch in sensor_channels:
            sig = sample[ch].flatten()
            N = len(sig)
            
            # --- Time-Domain Features ---
            sig_mean = float(np.mean(sig))
            sig_std = float(np.std(sig))
            sig_rms = float(np.sqrt(np.mean(sig**2)))
            sig_peak = float(np.max(np.abs(sig)))
            sig_kurt = float(kurtosis(sig))
            sig_crest = float(sig_peak / (sig_rms + 1e-9))
            
            row_dict[f'{ch}_mean'] = sig_mean
            row_dict[f'{ch}_std'] = sig_std
            row_dict[f'{ch}_rms'] = sig_rms
            row_dict[f'{ch}_peak'] = sig_peak
            row_dict[f'{ch}_kurtosis'] = sig_kurt
            row_dict[f'{ch}_crest_factor'] = sig_crest
            
            # --- Frequency-Domain Features (FFT) ---
            # Compute one-sided FFT spectrum
            fft_vals = np.abs(np.fft.rfft(sig - sig_mean))
            freqs = np.fft.rfftfreq(N, d=1.0 / sampling_rate)
            
            # Dominant frequency & spectral energy
            dom_freq = float(freqs[np.argmax(fft_vals)]) if len(fft_vals) > 0 else 0.0
            spectral_energy = float(np.sum(fft_vals**2) / N)
            
            row_dict[f'{ch}_dom_freq'] = dom_freq
            row_dict[f'{ch}_spec_energy'] = spectral_energy
            
        rows.append(row_dict)
        
    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION ENGINE REPORT")
    print("=" * 60)
    print(f"Total raw runs in file: {len(data)}")
    print(f"Dropped runs without measured wear labels: {len(dropped_runs)}")
    print(f"Retained valid runs for Machine Learning: {len(df)}")
    print(f"Total feature columns generated: {df.shape[1] - 8} sensor features (+ 8 metadata/targets)")
    print("\nWear Class Distribution (Ground Truth):")
    print(df['wear_class'].value_counts())
    print("\nFlank Wear (VB) Ground Truth Stats (mm):")
    print(df['VB'].describe())
    
    # Save extracted dataset
    output_csv = 'extracted_features.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nExtracted dataset saved to '{output_csv}' ({df.shape[0]} rows x {df.shape[1]} columns)")
    return df, dropped_runs

if __name__ == '__main__':
    extract_features_from_mill()
