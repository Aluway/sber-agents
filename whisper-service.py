from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import tempfile
import os

app = Flask(__name__)

# Загрузка модели при старте (base - хороший баланс скорость/качество)
print('Loading Whisper model...')
model = WhisperModel('base', device='cpu', compute_type='int8')
print('Model loaded!')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Получаем аудио файл
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        audio_file = request.files['file']
        
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg') as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Транскрибация
            segments, info = model.transcribe(tmp_path, language='ru')
            text = ' '.join([segment.text for segment in segments])
            
            return jsonify({
                'text': text,
                'language': info.language,
                'duration': info.duration
            })
        finally:
            # Удаляем временный файл
            os.unlink(tmp_path)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'whisper-base'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

