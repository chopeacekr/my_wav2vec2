# Wav2Vec2 STT Server

한국어 음성 인식을 위한 Wav2Vec2 기반 STT 서버

## 🎯 특징

- **모델**: `kresnik/wav2vec2-large-xlsr-korean` (한국어 Fine-tuned)
- **아키텍처**: Transformer Encoder + CTC Decoding
- **포트**: 8400
- **샘플레이트**: 16kHz
- **GPU/CPU**: 자동 감지 및 선택

## 📦 설치

### 1. 시스템 의존성

```bash
# Ubuntu/Debian
sudo apt-get install -y ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile
```

### 2. Python 의존성

```bash
# UV 사용
cd my_wav2vec2
uv sync

# 또는 pip
pip install -e .
```

## 🚀 실행

### 서버 시작

```bash
# 방법 1: 직접 실행
uv run python server_stt.py

# 방법 2: uvicorn 사용
uv run uvicorn server_stt:app --host 0.0.0.0 --port 8400

# 백그라운드 실행
nohup uv run python server_stt.py > wav2vec2_stt.log 2>&1 &
```

### 서버 확인

```bash
# 헬스 체크
curl http://localhost:8400/health

# 예상 응답:
# {
#   "status": "ok",
#   "model_loaded": true,
#   "processor_loaded": true,
#   "device": "cpu",
#   "model_id": "kresnik/wav2vec2-large-xlsr-korean"
# }
```

## 📡 API 사용법

### 1. 파일 업로드 방식

```bash
curl -X POST http://localhost:8400/transcribe \
  -F "file=@test_audio.wav" \
  -F "lang=KR"
```

**응답**:
```json
{
  "text": "안녕하세요",
  "language": "KR",
  "model": "kresnik/wav2vec2-large-xlsr-korean"
}
```

### 2. Python 클라이언트

```python
import requests

# 파일 업로드
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8400/transcribe",
        files={"file": f},
        data={"lang": "KR"}
    )

result = response.json()
print(result["text"])
```

## 🔧 모델 정보

### Wav2Vec2 아키텍처

```
Raw Audio (16kHz)
    ↓
Feature Encoder (CNN)
    ↓
Transformer Encoder (12 layers)
    ↓
CTC Head
    ↓
Text Output
```

### 모델 특징

- **사전 학습**: 다국어 음성 데이터 (XLSR-53)
- **Fine-tuning**: 한국어 데이터셋 (KSS, AI Hub)
- **정확도**: WER ~10-15% (깨끗한 음성)
- **속도**: CPU ~1-2초, GPU ~0.3-0.5초 (10초 오디오)

### 지원 언어

현재는 한국어만 지원:
- `KR`: Korean (kresnik/wav2vec2-large-xlsr-korean)

향후 추가 가능:
- `EN`: English (facebook/wav2vec2-base-960h)
- `JA`: Japanese
- `ZH`: Chinese

## 📊 성능

### 벤치마크 (10초 오디오)

| 환경 | 추론 시간 | 메모리 |
|------|----------|--------|
| CPU (i7-1255U) | ~1.5초 | ~2GB |
| GPU (CUDA) | ~0.4초 | ~3GB |

### WER (Word Error Rate)

| 환경 | WER |
|------|-----|
| 깨끗한 음성 | 10-15% |
| 배경 소음 | 20-30% |
| 음악 포함 | 40%+ |

## 🐛 문제 해결

### 1. 모델 다운로드 실패

**증상**:
```
OSError: Can't load tokenizer for 'kresnik/wav2vec2-large-xlsr-korean'
```

**해결**:
```bash
# 인터넷 연결 확인
ping huggingface.co

# 캐시 삭제
rm -rf ~/.cache/huggingface

# 재시도
uv run python server_stt.py
```

### 2. GPU 인식 안 됨

**증상**:
```
사용 디바이스: cpu
```

**해결**:
```bash
# CUDA 설치 확인
nvidia-smi

# PyTorch CUDA 버전 설치
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. 메모리 부족

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결**:
- CPU 모드로 전환
- 배치 크기 줄이기
- 오디오 길이 제한 (30초 이하 권장)

### 4. 오디오 포맷 에러

**증상**:
```
ValueError: Audio file could not be loaded
```

**해결**:
```bash
# ffmpeg로 변환
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Python으로 변환
import librosa
audio, sr = librosa.load("input.mp3", sr=16000, mono=True)
```

## 🔄 업데이트

### 모델 업데이트

```python
# server_stt.py에서 모델 변경
SUPPORTED_LANGUAGES = {
    "KR": {
        "model_id": "new-korean-model",  # 여기 수정
        "name": "Korean",
        "sample_rate": 16000
    }
}
```

### 새 언어 추가

```python
# server_stt.py
SUPPORTED_LANGUAGES = {
    "KR": {...},
    "EN": {  # 새 언어 추가
        "model_id": "facebook/wav2vec2-base-960h",
        "name": "English",
        "sample_rate": 16000
    }
}
```

## 📚 참고 자료

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [Hugging Face Model](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean)
- [Transformers Documentation](https://huggingface.co/docs/transformers/model_doc/wav2vec2)

## 📞 문의

- 작성자: Peace Cho
- 이메일: chopeacekr@gmail.com
- GitHub: https://github.com/chopeace/my-voice-lab

---

**Made with ❤️ for Korean STT**