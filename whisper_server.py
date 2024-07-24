from flask import Flask, request, jsonify
import whisper
import io
import numpy as np
from scipy.io import wavfile

app = Flask(__name__)
model = whisper.load_model("base")

@app.route('/transcribe', methods=['POST'])

def transcribe_audio():

	transcript = ""

	try: 
		audio_data = request.data
		#print("Received audio data")
		
		audio_io = io.BytesIO(audio_data)
		sample_rate, audio_np = wavfile.read(audio_io)
		#print("Audio data read into NumPy array")
		
		if len(audio_np.shape) > 1:
			audio_np = np.mean(audio_np, axis = 1)
		#print("Audio data converted to mono")	
			
		audio_np = audio_np.astype(np.float32)
		audio_np = audio_np / np.max(np.abs(audio_np))	
		#print("Audio data normalized and converted to float32")
		
		
		result = model.transcribe(audio_np)
		transcript = transcript + result["text"]
		#print("Transcription result: ", result["text"])
		print(transcript)
		return jsonify(result)
		
	except Exception as e:
		print("Error processing audio data:", e)
		return jsonify({"error:": str(e)}), 500
			
		
'''
		with wave.open(io.BytesIO(audio_data), 'wb') as wf:
			wf.setnchannels(1)
			wf.setsampwidth(2)
			wf.setframerate(16000)
			wf.writeframes(audio_data)
			
		result = model.transcribe(io.BytesIO(audio_data))
		return jsonify(result)
		'''
		
if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000)
