"""
Wav2Vec2 STT Server (CPU Optimized)
FastAPI 기반 한국어 음성 인식 서버

모델: kresnik/wav2vec2-large-xlsr-korean
포트: 8400
최적화: CPU 전용 (멀티스레딩 지원)
"""

import io
import logging
import os
from typing import Dict

import librosa
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ==========================================
# CPU 최적화 설정
# ==========================================

# CPU 스레드 수 설정 (환경 변수로 제어 가능)
CPU_THREADS = int(os.getenv("OMP_NUM_THREADS", "4"))

# PyTorch CPU 스레드 설정
torch.set_num_threads(CPU_THREADS)

# MKL 스레드 설정 (Intel CPU 최적화)
if os.getenv("MKL_NUM_THREADS") is None:
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="Wav2Vec2 STT Server (CPU)", version="1.0.0")

# 전역 변수
processor = None
model = None
device = None

# 지원 언어 (확장 가능)
SUPPORTED_LANGUAGES = {
    "KR": {
        "model_id": "kresnik/wav2vec2-large-xlsr-korean",
        "name": "Korean",
        "sample_rate": 16000
    },
    # 향후 다른 언어 추가 가능
    # "EN": {
    #     "model_id": "facebook/wav2vec2-base-960h",
    #     "name": "English",
    #     "sample_rate": 16000
    # }
}

DEFAULT_LANGUAGE = "KR"


@app.on_event("startup")
async def load_model():
    """서버 시작 시 모델 로드 (CPU 최적화)"""
    global processor, model, device
    
    logger.info("🚀 Wav2Vec2 STT Server 시작 중... (CPU Optimized)")
    
    # ⭐ CPU 전용 디바이스 설정
    device = torch.device("cpu")
    logger.info(f"📱 사용 디바이스: {device}")
    logger.info(f"🧵 CPU 스레드 수: {CPU_THREADS}")
    
    # CPU 정보 출력
    logger.info(f"💻 CPU 코어 수: {os.cpu_count()}")
    logger.info(f"🔧 PyTorch 스레드: {torch.get_num_threads()}")
    
    # 기본 한국어 모델 로드
    model_id = SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]["model_id"]
    logger.info(f"📦 모델 로딩 중: {model_id}")
    logger.info(f"⏳ 첫 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다 (~1.2GB)")
    
    try:
        # Processor (토크나이저) 로드
        logger.info("📥 Processor 다운로드 중...")
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        logger.info("✅ Processor 로드 완료")
        
        # ⭐ 모델 로드 (CPU 최적화)
        logger.info("📥 모델 다운로드 중... (용량 큼, 시간 소요)")
        model = Wav2Vec2ForCTC.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # ⭐ CPU는 float32 사용 (float16 불안정)
            low_cpu_mem_usage=True,     # ⭐ 메모리 사용량 최적화
        )
        
        # CPU로 이동 (이미 CPU지만 명시적으로)
        model.to(device)
        
        # ⭐ 평가 모드 (드롭아웃 비활성화)
        model.eval()
        
        logger.info("✅ 모델 로드 완료")
        
        # 메모리 정보 출력
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"💾 메모리 사용량: {memory_info.rss / 1024 / 1024:.1f} MB")
        except ImportError:
            logger.info("💡 psutil 설치 시 메모리 정보 확인 가능: pip install psutil")
        
        logger.info(f"🎉 Wav2Vec2 STT Server 준비 완료! (포트: 8400)")
        logger.info(f"📊 예상 성능: 10초 오디오 → 약 {CPU_THREADS}코어 기준 1.5-2초 처리")
        
    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        logger.error(f"💡 해결 방법:")
        logger.error(f"   1. 인터넷 연결 확인 (Hugging Face 다운로드 필요)")
        logger.error(f"   2. 캐시 삭제: rm -rf ~/.cache/huggingface")
        logger.error(f"   3. 메모리 확인: 최소 4GB RAM 권장")
        raise


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "Wav2Vec2 STT Server (CPU Optimized)",
        "version": "1.0.0",
        "status": "running",
        "model": SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]["model_id"],
        "device": str(device),
        "cpu_threads": CPU_THREADS,
        "cpu_cores": os.cpu_count(),
        "supported_languages": list(SUPPORTED_LANGUAGES.keys())
    }


@app.get("/health")
async def health_check() -> Dict:
    """헬스 체크 엔드포인트"""
    is_ready = model is not None and processor is not None
    
    health_info = {
        "status": "ok" if is_ready else "loading",
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "device": str(device),
        "cpu_threads": CPU_THREADS,
        "cpu_cores": os.cpu_count(),
        "model_id": SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]["model_id"]
    }
    
    # 메모리 정보 추가 (psutil 있을 경우)
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        health_info["memory_mb"] = round(memory_info.rss / 1024 / 1024, 1)
    except ImportError:
        pass
    
    return health_info


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    lang: str = DEFAULT_LANGUAGE
) -> JSONResponse:
    """
    오디오 파일을 텍스트로 변환 (CPU 최적화)
    
    Args:
        file: 오디오 파일 (WAV, MP3, FLAC 등)
        lang: 언어 코드 (기본: KR)
    
    Returns:
        JSONResponse: {"text": "변환된 텍스트", "language": "KR"}
    """
    logger.info(f"📝 STT 요청 받음 (파일: {file.filename}, 언어: {lang})")
    
    # 모델 로드 확인
    if model is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="모델이 아직 로드되지 않았습니다. 잠시 후 다시 시도하세요."
        )
    
    # 언어 지원 확인
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 언어입니다. 지원 언어: {list(SUPPORTED_LANGUAGES.keys())}"
        )
    
    try:
        # 오디오 파일 읽기
        audio_bytes = await file.read()
        logger.info(f"📦 오디오 데이터 크기: {len(audio_bytes)} bytes")
        
        # 오디오 전처리
        target_sr = SUPPORTED_LANGUAGES[lang]["sample_rate"]
        audio, sample_rate = librosa.load(
            io.BytesIO(audio_bytes),
            sr=target_sr,
            mono=True
        )
        logger.info(f"🎵 오디오 로드 완료 (샘플레이트: {sample_rate}Hz, 길이: {len(audio)} samples)")
        
        # 오디오가 너무 짧으면 에러
        if len(audio) < 1600:  # 0.1초 미만
            raise HTTPException(
                status_code=400,
                detail="오디오가 너무 짧습니다 (최소 0.1초 필요)"
            )
        
        # ⭐ CPU 최적화된 STT 추론
        import time
        start_time = time.time()
        
        with torch.no_grad():  # ⭐ 그래디언트 계산 비활성화 (메모리 절약)
            # 입력 준비
            input_values = processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_values
            
            # CPU로 이동 (이미 CPU지만 명시적으로)
            input_values = input_values.to(device)
            
            # ⭐ 모델 추론 (CPU)
            logits = model(input_values).logits
            
            # 디코딩 (Greedy Decoding)
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # 텍스트 변환
            transcription = processor.batch_decode(predicted_ids)[0]
        
        # 처리 시간 측정
        elapsed_time = time.time() - start_time
        logger.info(f"✅ STT 변환 완료: '{transcription}' (처리 시간: {elapsed_time:.2f}초)")
        
        return JSONResponse(content={
            "text": transcription,
            "language": lang,
            "model": SUPPORTED_LANGUAGES[lang]["model_id"],
            "processing_time_seconds": round(elapsed_time, 2),
            "device": "cpu",
            "cpu_threads": CPU_THREADS
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ STT 처리 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"STT 처리 중 오류 발생: {str(e)}"
        )


@app.post("/transcribe_bytes")
async def transcribe_bytes(
    audio_bytes: bytes,
    lang: str = DEFAULT_LANGUAGE,
    sample_rate: int = 16000
) -> Dict:
    """
    오디오 바이트를 직접 받아 텍스트로 변환 (내부 API, CPU 최적화)
    
    Args:
        audio_bytes: WAV 오디오 바이트
        lang: 언어 코드
        sample_rate: 샘플레이트
    
    Returns:
        Dict: {"text": "변환된 텍스트", "processing_time_seconds": 1.5}
    """
    logger.info(f"📝 STT 바이트 요청 (크기: {len(audio_bytes)} bytes)")
    
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="모델 로드 중")
    
    try:
        import time
        start_time = time.time()
        
        # 오디오 로드
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        
        # 모노 변환 (스테레오일 경우)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # 리샘플링 (필요 시)
        target_sr = SUPPORTED_LANGUAGES[lang]["sample_rate"]
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # ⭐ STT 추론 (CPU 최적화)
        with torch.no_grad():
            input_values = processor(
                audio,
                sampling_rate=target_sr,
                return_tensors="pt"
            ).input_values.to(device)
            
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ STT 완료: '{transcription}' ({elapsed_time:.2f}초)")
        
        return {
            "text": transcription,
            "processing_time_seconds": round(elapsed_time, 2)
        }
        
    except Exception as e:
        logger.error(f"❌ STT 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════╗
    ║   Wav2Vec2 STT Server (CPU Optimized)     ║
    ║   포트: 8400                              ║
    ║   모델: kresnik/wav2vec2-large-xlsr-korean║
    ║   디바이스: CPU                           ║
    ║   스레드: {}                            ║
    ╚═══════════════════════════════════════════╝
    """.format(CPU_THREADS))
    
    print(f"💡 CPU 성능 최적화 팁:")
    print(f"   export OMP_NUM_THREADS={os.cpu_count()}  # CPU 코어 수만큼")
    print(f"   export MKL_NUM_THREADS={os.cpu_count()}")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8400,
        log_level="info"
    )