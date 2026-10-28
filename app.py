import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import kurtosis
from flask import Flask, render_template, jsonify, request, send_file, send_from_directory

app = Flask(__name__, static_folder='static', template_folder='templates')

# --- Load Dataset and Models at Startup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models_and_plots')
MAT_FILE = os.path.join(BASE_DIR, 'mill.mat')
GDRIVE_FILE_ID="1kcXiGrhhy6Dxv8w3nZXPW9GWkrIu6t7o"

print(f"Loading '{MAT_FILE}'...")
mat_data = sio.loadmat(MAT_FILE)['mill'][0]

print("Loading trained models and feature scaler...")
scaler = joblib.load(os.path.join(MODELS_DIR, 'feature_scaler.joblib'))
feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))

trained_models = {
    'ANN': {
        'name': 'Artificial Neural Network (MLP)',
        'model': joblib.load(os.path.join(MODELS_DIR, 'artificial_neural_network.joblib')),
        'scaled': True,
        'r2': '94.12%',
        'mae': '0.053 mm'
    },
    'SVR': {
        'name': 'Support Vector Regressor (SVR)',
        'model': joblib.load(os.path.join(MODELS_DIR, 'support_vector_regressor.joblib')),
        'scaled': True,
        'r2': '84.10%',
        'mae': '0.082 mm'
    },
    'RF': {
        'name': 'Random Forest Regressor',
        'model': joblib.load(os.path.join(MODELS_DIR, 'random_forest_regressor.joblib')),
        'scaled': False,
        'r2': '69.32%',
        'mae': '0.106 mm'
    },
    'LR': {
        'name': 'Linear Regression (Baseline)',
        'model': joblib.load(os.path.join(MODELS_DIR, 'linear_regression.joblib')),
        'scaled': True,
        'r2': '86.15%',
        'mae': '0.087 mm'
    }
}

# Pre-parse experimental cases from mill.mat
cases_meta = {}
case_runs = {}

for idx, run in enumerate(mat_data):
    case_id = int(run['case'].squeeze())
    cut_id = int(run['run'].squeeze())
    doc = float(run['DOC'].squeeze())
    feed = float(run['feed'].squeeze())
    mat_type = int(run['material'].squeeze())
    mat_name = 'Cast Iron' if mat_type == 1 else 'Stainless Steel J45'
    
    # Check if signal is valid (not corrupted DAQ spike)
    corrupted = False
    for ch in ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']:
        sig_test = run[ch].flatten()
        if np.max(np.abs(sig_test)) > 15.0:
            corrupted = True
            break
    if corrupted:
        continue

    vb_raw = run['VB']
    vb_val = float(vb_raw.squeeze()) if vb_raw.size > 0 and not np.isnan(float(vb_raw.squeeze())) else None
    
    if case_id not in cases_meta:
        cases_meta[case_id] = {
            'case_id': case_id,
            'material': mat_name,
            'DOC': doc,
            'feed': feed,
            'total_cuts': 0,
            'cuts_with_wear': 0
        }
        case_runs[case_id] = []
        
    cases_meta[case_id]['total_cuts'] += 1
    if vb_val is not None:
        cases_meta[case_id]['cuts_with_wear'] += 1
        
    case_runs[case_id].append({
        'run_idx': idx,
        'cut_id': cut_id,
        'actual_vb': vb_val,
        'DOC': doc,
        'feed': feed,
        'material': mat_type,
        'material_name': mat_name
    })

# In-memory session telemetry history
telemetry_history = []
alert_log = []

def extract_features_from_sample(sample_record, cut_counter):
    """
    Computes identical 52 features as used during training.
    """
    sensor_channels = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']
    sampling_rate = 250.0  # Hz
    
    feat_dict = {
        'DOC': float(sample_record['DOC'].squeeze()),
        'feed': float(sample_record['feed'].squeeze()),
        'material': int(sample_record['material'].squeeze()),
        'cut': int(cut_counter)
    }
    
    for ch in sensor_channels:
        sig = sample_record[ch].flatten()
        N = len(sig)
        mean_val = float(np.mean(sig))
        std_val = float(np.std(sig))
        rms_val = float(np.sqrt(np.mean(sig**2)))
        peak_val = float(np.max(np.abs(sig)))
        kurt_val = float(kurtosis(sig))
        crest_val = float(peak_val / (rms_val + 1e-9))
        
        fft_vals = np.abs(np.fft.rfft(sig - mean_val))
        freqs = np.fft.rfftfreq(N, d=1.0 / sampling_rate)
        dom_freq = float(freqs[np.argmax(fft_vals)]) if len(fft_vals) > 0 else 0.0
        spec_energy = float(np.sum(fft_vals**2) / N)
        
        feat_dict[f'{ch}_mean'] = mean_val
        feat_dict[f'{ch}_std'] = std_val
        feat_dict[f'{ch}_rms'] = rms_val
        feat_dict[f'{ch}_peak'] = peak_val
        feat_dict[f'{ch}_kurtosis'] = kurt_val
        feat_dict[f'{ch}_crest_factor'] = crest_val
        feat_dict[f'{ch}_dom_freq'] = dom_freq
        feat_dict[f'{ch}_spec_energy'] = spec_energy
        
    feat_df = pd.DataFrame([feat_dict])[feature_names]
    return feat_df, feat_dict

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metadata')
def get_metadata():
    return jsonify({
        'cases': cases_meta,
        'models': {
            k: {
                'name': v['name'],
                'r2': v['r2'],
                'mae': v['mae']
            } for k, v in trained_models.items()
        },
        'thresholds': {
            'healthy_max': 0.20,
            'warning_max': 0.40
        }
    })

@app.route('/api/process_cut', methods=['POST'])
def process_cut():
    """
    Feeds a real continuous cut from the recorded dataset into the feature engine
    and ML model, evaluates wear thresholds, and generates telemetry.
    """
    req = request.get_json() or {}
    case_id = int(req.get('case_id', 1))
    cut_step = int(req.get('cut_step', 0))
    selected_model_key = req.get('model_key', 'ANN')
    
    if case_id not in case_runs or len(case_runs[case_id]) == 0:
        return jsonify({'error': f'Case {case_id} not found'}), 400
        
    cuts_list = case_runs[case_id]
    total_cuts = len(cuts_list)
    
    if cut_step >= total_cuts:
        return jsonify({
            'status': 'completed',
            'message': f'All {total_cuts} cuts for Case {case_id} completed.',
            'total_cuts': total_cuts
        })
        
    cut_info = cuts_list[cut_step]
    raw_sample = mat_data[cut_info['run_idx']]
    
    # Extract features using verified engine
    feat_df, feat_dict = extract_features_from_sample(raw_sample, cut_info['cut_id'])
    
    # Model Inference
    model_entry = trained_models.get(selected_model_key, trained_models['ANN'])
    model = model_entry['model']
    
    if model_entry['scaled']:
        feat_input = scaler.transform(feat_df)
    else:
        feat_input = feat_df.values
        
    pred_vb = float(model.predict(feat_input)[0])
    pred_vb = max(0.0, pred_vb)  # Tool wear cannot physically be negative
    
    actual_vb = cut_info['actual_vb']
    error = abs(pred_vb - actual_vb) if actual_vb is not None else None
    
    # Condition Classification
    if pred_vb < 0.20:
        condition = 'Healthy'
        condition_level = 'success'
    elif pred_vb <= 0.40:
        condition = 'Warning'
        condition_level = 'warning'
    else:
        condition = 'Critical'
        condition_level = 'danger'
        
    # Generate Alerts when thresholds are crossed
    alert = None
    if condition == 'Warning':
        alert = {
            'level': 'warning',
            'title': 'Tool Wear Warning',
            'message': f"Pass #{cut_info['cut_id']}: Predicted flank wear ({pred_vb:.3f} mm) exceeded Warning limit (0.20 mm). Frictional contact on tool flank is increasing.",
            'cut': cut_info['cut_id']
        }
        alert_log.append(alert)
    elif condition == 'Critical':
        alert = {
            'level': 'danger',
            'title': 'CRITICAL TOOL REPLACEMENT REQUIRED',
            'message': f"Pass #{cut_info['cut_id']}: Predicted flank wear ({pred_vb:.3f} mm) exceeded Critical Limit (0.40 mm)! Stop machining to prevent tool fracture and workpiece damage.",
            'cut': cut_info['cut_id']
        }
        alert_log.append(alert)
        
    # Real Downsampled Sensor Waveforms (200 points per channel for smooth 60fps charting)
    ds_factor = 45
    waveforms = {
        'smcDC': raw_sample['smcDC'].flatten()[::ds_factor].tolist(),
        'smcAC': raw_sample['smcAC'].flatten()[::ds_factor].tolist(),
        'vib_spindle': raw_sample['vib_spindle'].flatten()[::ds_factor].tolist(),
        'vib_table': raw_sample['vib_table'].flatten()[::ds_factor].tolist(),
        'AE_spindle': raw_sample['AE_spindle'].flatten()[::ds_factor].tolist(),
        'AE_table': raw_sample['AE_table'].flatten()[::ds_factor].tolist(),
        'time_axis': np.round(np.linspace(0, 36.0, len(raw_sample['smcDC'].flatten()[::ds_factor])), 2).tolist()
    }
    
    # Record in session telemetry history
    telemetry_entry = {
        'timestamp_cut': cut_info['cut_id'],
        'case_id': case_id,
        'material': cut_info['material_name'],
        'DOC': cut_info['DOC'],
        'feed': cut_info['feed'],
        'model_used': model_entry['name'],
        'predicted_vb': round(pred_vb, 4),
        'actual_vb': round(actual_vb, 4) if actual_vb is not None else 'Unmeasured',
        'error_mm': round(error, 4) if error is not None else 'N/A',
        'condition': condition,
        'smcDC_mean': round(feat_dict['smcDC_mean'], 3),
        'smcDC_rms': round(feat_dict['smcDC_rms'], 3),
        'vib_spindle_rms': round(feat_dict['vib_spindle_rms'], 3),
        'AE_spindle_rms': round(feat_dict['AE_spindle_rms'], 3)
    }
    telemetry_history.append(telemetry_entry)
    
    return jsonify({
        'status': 'success',
        'case_id': case_id,
        'cut_step': cut_step,
        'next_step': cut_step + 1,
        'total_cuts': total_cuts,
        'cut_id': cut_info['cut_id'],
        'DOC': cut_info['DOC'],
        'feed': cut_info['feed'],
        'material_name': cut_info['material_name'],
        'predicted_vb': pred_vb,
        'actual_vb': actual_vb,
        'error': error,
        'condition': condition,
        'condition_level': condition_level,
        'alert': alert,
        'waveforms': waveforms,
        'features': {
            'smcDC_mean': feat_dict['smcDC_mean'],
            'smcDC_rms': feat_dict['smcDC_rms'],
            'vib_spindle_rms': feat_dict['vib_spindle_rms'],
            'AE_spindle_rms': feat_dict['AE_spindle_rms'],
            'smcAC_std': feat_dict['smcAC_std']
        }
    })

@app.route('/api/reset', methods=['POST'])
def reset_session():
    telemetry_history.clear()
    alert_log.clear()
    return jsonify({'status': 'reset_successful'})

@app.route('/api/history')
def get_history():
    return jsonify({
        'history': telemetry_history,
        'alerts': alert_log
    })

@app.route('/api/export_history_csv')
def export_history_csv():
    if not telemetry_history:
        return "No telemetry data recorded yet.", 400
    export_df = pd.DataFrame(telemetry_history)
    csv_path = os.path.join(BASE_DIR, 'telemetry_session_export.csv')
    export_df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True, download_name='telemetry_session_export.csv')

@app.route('/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(MODELS_DIR, filename)

if __name__ == '__main__':
    print("Starting Tool Wear Continuous Monitoring Web Server on port 5000...")
    app.run(host='127.0.0.1', port=5000, debug=False)
