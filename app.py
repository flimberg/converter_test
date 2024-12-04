from flask import Flask, request, render_template, send_file, jsonify
from obspy import read
from obspy.signal.filter import bandpass
from scipy.io.wavfile import write
import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from matplotlib import mlab

# Set global font sizes
plt.rcParams['axes.labelsize'] = 22   # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 22  # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 22  # Font size for y-axis tick labels
plt.rcParams['legend.fontsize'] = 15  # Font size for legend


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot_waveform():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save the uploaded MiniSEED file
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        # Get filter parameters
        low_cutoff = float(request.form.get('low_cutoff', 0))  # Default: no lower limit
        high_cutoff = float(request.form.get('high_cutoff', 0))  # Default: no upper limit

        # Read MiniSEED file using ObsPy
        st = read(input_path)
        st = st.detrend("demean")
        tr = st[0]  # Use the first trace in the stream

        # Apply bandpass filter if cutoffs are provided
        if low_cutoff > 0 or high_cutoff > 0:
            tr.data = bandpass(tr.data, low_cutoff, high_cutoff, df=tr.stats.sampling_rate, corners=4, zerophase=True)

        # Generate waveform plot
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        axs[0].plot(tr.times(), tr.data)
        axs[0].set_title('Waveform')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_ylim(-np.max(np.abs(tr.data)),np.max(np.abs(tr.data)))

        # Generate spectrogram
        specgram_infra, freqs, times = mlab.specgram(tr.data, Fs=tr.stats.sampling_rate, mode='magnitude')
        axs[1].pcolormesh(times, freqs, np.log(specgram_infra), shading='gouraud', cmap="turbo", vmin=-1)
        axs[1].set_title('Spectrogram')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Frequency (Hz)')

        # Save the plot to a BytesIO buffer
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return jsonify({"plot": plot_data, "filename": file.filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/convert', methods=['POST'])
def convert_mseed_to_wav():
    filename = request.form.get('filename')
    pitch = int(request.form.get('pitch', 1))  # Default pitch is 1
    low_cutoff = float(request.form.get('low_cutoff', 0))  # Default: no lower limit
    high_cutoff = float(request.form.get('high_cutoff', 0))  # Default: no upper limit
    if not filename:
        return "No file specified", 400

    try:
        # Load the MiniSEED file
        if filename == "testdata.mseed":
            input_path = os.path.join(os.path.dirname(__file__), 'testdata.mseed')
        else:
            input_path = os.path.join(UPLOAD_FOLDER, filename)
        
        st = read(input_path)
        tr = st[0]  # Use the first trace in the stream

        # Apply bandpass filter if cutoffs are provided
        if low_cutoff > 0 or high_cutoff > 0:
            tr.data = bandpass(tr.data, low_cutoff, high_cutoff, df=tr.stats.sampling_rate, corners=4, zerophase=True)

        # Adjust sampling rate based on pitch
        sampling_rate = tr.stats.sampling_rate * pitch
        data = tr.data

        # Normalize and convert data for WAV format
        data = np.int16(data / np.max(np.abs(data)) * 32767)

        # Save as WAV
        output_filename = os.path.splitext(filename)[0] + f'_pitch_{pitch}.wav'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        write(output_path, int(sampling_rate), data)
        
        # Return WAV file as downloadable response
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"Error during conversion: {e}", 500

@app.route('/plot/testdata', methods=['POST'])
def plot_testdata():
    try:
        # Load test data file
        input_path = os.path.join(os.path.dirname(__file__), 'testdata.mseed')

        # Get filter parameters
        data = request.get_json()
        low_cutoff = float(data.get('low_cutoff', 0))  # Default: no lower limit
        high_cutoff = float(data.get('high_cutoff', 0))  # Default: no upper limit

        # Read MiniSEED file using ObsPy
        st = read(input_path)
        st = st.detrend("demean")
        tr = st[0]  # Use the first trace in the stream

        # Apply bandpass filter if cutoffs are provided
        if low_cutoff > 0 or high_cutoff > 0:
            tr.data = bandpass(tr.data, low_cutoff, high_cutoff, df=tr.stats.sampling_rate, corners=4, zerophase=True)
            
        
        # Generate waveform plot
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        axs[0].plot(tr.times(), tr.data)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_ylim(-np.max(np.abs(tr.data)),np.max(np.abs(tr.data)))

        # Generate spectrogram
        specgram_infra, freqs, times = mlab.specgram(tr.data, Fs=tr.stats.sampling_rate, mode='magnitude')
        axs[1].pcolormesh(times, freqs, np.log(specgram_infra), shading='gouraud', cmap="turbo", vmin=-1)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Frequency (Hz)')

        # Save the plot to a BytesIO buffer
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return jsonify({"plot": plot_data, "filename": "testdata.mseed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
