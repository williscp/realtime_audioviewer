from flask import Flask, Response,render_template
import pyaudio
import threading
import argparse
import matplotlib.pyplot as plt
import time
import io
import numpy as np
import cv2

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
chunk = 1024
bitsPerSample = 16
channels = 2
wav_header = genHeader(rate, bitsPerSample, channels)

stream = audio1.open(format=format, channels=channels,
    rate=rate, input=True, input_device_index=8,
    frames_per_buffer=chunk)

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

    frames = []
    while True:
        data = stream.read(chunk)
        frames.append(np.fromstring(data,  dtype=np.int16))
        if total > bucket_size:
            pass
        total += 1
        with lock:
            outputBucket = frames

def generate():
    global outputBucket, lock
    while True:
        with lock:
            if outputBucket is None:
                continue
            #print(outputBucket)
            #(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            #if not flag:
            #continue

            plt.figure()
            plt.plot(np.mean(outputBucket, axis=1))
            plt.savefig("test.jpg")
            plt.close()
            img = cv2.imread("test.jpg")
            #encodedImage = buf.read()
            (flag, encodedImage) = cv2.imencode(".jpg", img)

            #im = Image.open(buf)
            if not flag:
                continue
            #im.show()

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
        bytearray(encodedImage) + b'\r\n')

@app.route('/audio_feed')
def audio_feed():

	return Response(generate(),
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
