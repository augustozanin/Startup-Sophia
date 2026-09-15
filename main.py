"""
Sophia - Backend de transcrição (100% local, sem SaaS externo)
------------------------------------------------------------------
Recebe blocos de áudio do navegador via WebSocket e transcreve
localmente usando faster-whisper (rodando na CPU). Nenhum áudio
sai da máquina onde este backend está rodando.

Rodar com:
    uvicorn main:app --reload --port 8000

Na primeira execução, o faster-whisper baixa o modelo escolhido
(uma vez só, fica em cache local) — depois disso funciona 100%
offline.
"""

import asyncio
import io

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI(title="Sophia - Serviço de Transcrição (local)")

# Em produção, restrinja allow_origins ao domínio real do frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregado uma única vez, na subida do servidor.
# "small" = bom equilíbrio entre precisão e velocidade em CPU.
# Se estiver lento demais na sua máquina, troque por "base" ou "tiny".
# Se um dia migrar para uma máquina com GPU, troque device="cuda"
# e compute_type="float16", e pode subir para "medium"/"large-v3".
MODEL_SIZE = "small"
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

# Precisa bater com OVERLAP_MS/1000 no frontend (index.html).
# É o quanto de áudio repetido existe no início de cada bloco recebido.
OVERLAP_SECONDS = 1.0


def transcrever_bloco(audio_bytes: bytes) -> str:
    """Roda de forma síncrona (bloqueante) — por isso é chamada via
    asyncio.to_thread, para não travar o loop de eventos do FastAPI
    enquanto o Whisper processa o áudio."""
    audio_buffer = io.BytesIO(audio_bytes)
    segments, info = model.transcribe(
        audio_buffer,
        language="pt",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True,
    )
    segments = list(segments)  # força a geração agora, dentro da thread

    # Cada bloco recebido tem os primeiros OVERLAP_SECONDS repetidos do
    # bloco anterior (ver frontend). Descartamos palavras que caem nessa
    # janela para não duplicar texto já enviado antes.
    palavras = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            if w.start >= OVERLAP_SECONDS - 0.05:
                palavras.append(w.word)

    texto = "".join(palavras).strip()

    # --- LOG DE DIAGNÓSTICO (temporário, pra investigar) ---
    print(
        f"[debug] bytes_recebidos={len(audio_bytes)} "
        f"duracao_audio={info.duration:.2f}s "
        f"idioma_detectado={info.language} "
        f"confianca_idioma={info.language_probability:.2f} "
        f"n_segmentos={len(segments)} "
        f"texto={texto!r}"
    )

    return texto


@app.get("/health")
async def health():
    return {"status": "ok", "modelo": MODEL_SIZE}


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Cada mensagem binária recebida é um BLOCO DE ÁUDIO
            # COMPLETO e independente (ver frontend/index.html),
            # não um pedaço de um stream contínuo.
            audio_bytes = await websocket.receive_bytes()

            if not audio_bytes:
                continue

            try:
                texto = await asyncio.to_thread(transcrever_bloco, audio_bytes)
            except Exception as exc:  # noqa: BLE001
                await websocket.send_json(
                    {"type": "error", "message": f"Falha ao transcrever: {exc}"}
                )
                continue

            if texto:
                await websocket.send_json(
                    {"type": "transcript", "text": texto, "is_final": True}
                )

    except WebSocketDisconnect:
        pass