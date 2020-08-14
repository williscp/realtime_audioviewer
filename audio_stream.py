from flask import Flask, Response,render_template
import pyaudio
import threading
import argparse
import matplotlib.pyplot as plt
import time
import io
import numpy as np
import cv2
import librosa
from torchvision.transforms import ToPILImage

from inference import AudioProcessor
from config import config

outputBucket = None
lock = threading.Lock()

app = Flask(__name__)

def genHeader(sampleRate, bitsPerSample, channels):
    datasize = 2000*10**6
    o = bytes("RIFF",'ascii')                                               # (4byte) Marks file as RIFF
    o += (datasize + 36).to_bytes(4,'little')                               # (4byte) File size in bytes excluding this and RIFF marker
    o += bytes("WAVE",'ascii')                                              # (4byte) File type
    o += bytes("fmt ",'ascii')                                              # (4byte) Format Chunk Marker
    o += (16).to_bytes(4,'little')                                          # (4byte) Length of above format data
    o += (1).to_bytes(2,'little')                                           # (2byte) Format type (1 - PCM)
    o += (channels).to_bytes(2,'little')                                    # (2byte)
    o += (sampleRate).to_bytes(4,'little')                                  # (4byte)
    o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4,'little')  # (4byte)
    o += (channels * bitsPerSample // 8).to_bytes(2,'little')               # (2byte)
    o += (bitsPerSample).to_bytes(2,'little')                               # (2byte)
    o += bytes("data",'ascii')                                              # (4byte) Data Chunk Marker
    o += (datasize).to_bytes(4,'little')                                    # (4byte) Data size in bytes
    return o

audio1 = pyaudio.PyAudio()

rate = 48000
RECORD_SECONDS = 5

format = pyaudio.paInt16
chunk = 4096
bitsPerSample = 16
channels = 2
wav_header = genHeader(rate, bitsPerSample, channels)

stream = audio1.open(format=format, channels=channels,
    rate=rate, input=True, input_device_index=5,
    frames_per_buffer=chunk)

processor = AudioProcessor(config)

time.sleep(2.0)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def sound():

    global stream, rate, chunk, outputBucket, lock

    total = 0
    print("recording...")
    #frames = []
    #first_run = False
    bucket_size = rate / chunk

    frames = np.array([])
    while True:
        data = stream.read(chunk, exception_on_overflow=False)
        frames = np.concatenate((frames, np.fromstring(data,  dtype=np.int16)))
        if total > bucket_size:
            pass
        total += 1
        with lock:
            outputBucket = frames
            frames = frames[-480000:]

def generate_mel():
    global outputBucket, lock, rate, processor
    while True:
        with lock:
            if outputBucket is None:
                continue
            """
            S = librosa.core.stft(y=outputBucket[-48000:],
                                  n_fft=1200,
                                  win_length=int(0.025 * 48000),
                                  hop_length=int(0.01 * 48000))
            S = np.abs(S)
            mel_basis = librosa.filters.mel(sr=48000,
                                            n_fft=1200,
                                            n_mels=80,
                                            fmin=20,
                                            fmax=8000)
            # S = np.log10(np.dot(mel_basis, S))           # log mel spectrogram of utterances
            # S = librosa.core.power_to_dB(np.dot(mel_basis, S))
            S = np.dot(mel_basis, S)          # log mel spectrogram of utterances
            S = np.log10(S)           # log mel spectrogram of utterances
            S [S < - 40] = - 40
            S = S*20          # log mel spectrogram of utterances
            """

            S = librosa.feature.melspectrogram(y=outputBucket[-48000:],sr=rate,
                n_fft=1200,
                win_length=int(0.025 * 48000),
                hop_length=int(0.01 * 48000),
                n_mels=80,
                fmin=20,
                fmax=8000)

            S = librosa.core.power_to_db(S,ref=1.0, top_db=80)

            #output = processor.process(S)

            #img = output
            #img = np.array(topil(output[0].detach()))
            plt.imshow(S)
            plt.savefig("test.jpg")
            plt.close()
            img = cv2.imread("test.jpg")

            #encodedImage = buf.read()
            (flag, encodedImage) = cv2.imencode(".jpg", img)

            if not flag:
                continue
            #im.show()

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
        bytearray(encodedImage) + b'\r\n')

def generate_vis():
    global outputBucket, lock, rate, processor
    while True:
        with lock:
            if outputBucket is None:
                continue
            #print(outputBucket)
            #(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            #if not flag:
            #continue

            """
            S = librosa.feature.melspectrogram(y=outputBucket,sr=rate,
                n_fft=hp.data.nfft,
                win_length=int(hp.data.window * sr),
                hop_length=int(hp.data.hop * sr),
                n_mels=hp.data.nmels,
                fmin=20,
                fmax=8000)
            """

            # N x 1 x 20 x 80

            # L x C x T x F

            # 1 x 1 x 20 x 80


            # 19 * 480
            # 22000
            # 11000

            # 80 x L
            """
            S = librosa.core.stft(y=outputBucket[-9120:],
                                  n_fft=1200,
                                  win_length=int(0.025 * 48000),
                                  hop_length=int(0.01 * 48000))
            S = np.abs(S)
            mel_basis = librosa.filters.mel(sr=48000,
                                            n_fft=1200,
                                            n_mels=80,
                                            fmin=20,
                                            fmax=8000)
            # S = np.log10(np.dot(mel_basis, S))           # log mel spectrogram of utterances
            # S = librosa.core.power_to_dB(np.dot(mel_basis, S))
            S = np.dot(mel_basis, S)          # log mel spectrogram of utterances
            S = np.log10(S)           # log mel spectrogram of utterances
            S [S < - 40] = - 40
            S = S * 20          # log mel spectrogram of utterances

            # 19 * 480= 9120
            """
            S = librosa.feature.melspectrogram(y=outputBucket[-9360:],sr=rate,
                n_fft=1200,
                win_length=int(0.025 * 48000),
                hop_length=int(0.01 * 48000),
                n_mels=80,
                fmin=20,
                fmax=8000)

            S = librosa.core.power_to_db(S,ref=1.0, top_db=80)

            #plt.plot(outputBucket)

            output = processor.process(S)

            img = output
            #img = np.array(topil(output[0].detach()))
            """
            plt.imshow(S)
            plt.savefig("test.jpg")
            plt.close()
            img = cv2.imread("test.jpg")
            """
            #encodedImage = buf.read()
            img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            (flag, encodedImage) = cv2.imencode(".jpg", img)

            if not flag:
                continue
            #im.show()

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
        bytearray(encodedImage) + b'\r\n')


#@app.route('/audio_feed')
#def audio_feed():

#	return Response(generate_audio(),
#		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/mel_feed')
def mel_feed():

	return Response(generate_mel(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/vis_feed')
def vis_feed():

	return Response(generate_vis(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

#if __name__ == "__main__":
#    app.run(host='0.0.0.0', debug=True, threaded=True,port=5000)

if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	# start a thread that will perform motion detection
	t = threading.Thread(target=sound)
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
# release the video stream pointer
