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
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Sophia - Serviço de Transcrição (local)")

# Em produção, restrinja allow_origins ao domínio real do frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Conexão com o MongoDB Atlas ---
MONGODB_URI = os.getenv("MONGODB_URI")
mongo_client = AsyncIOMotorClient(MONGODB_URI) if MONGODB_URI else None
mongo_db = mongo_client["sophia-data"] if mongo_client else None
transcript_collection = mongo_db["transcript"] if mongo_db is not None else None

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

# --- Modo de diagnóstico: salva em disco o áudio exato recebido de cada
# bloco (já com overlap aplicado), para ouvir e confirmar se o problema
# está na captura/envio (frontend) ou na transcrição em si (Whisper).
SALVAR_AUDIO_DEBUG = True
PASTA_DEBUG = "debug_audio"
if SALVAR_AUDIO_DEBUG:
    os.makedirs(PASTA_DEBUG, exist_ok=True)


def transcrever_bloco(audio_bytes: bytes) -> str:
    """Roda de forma síncrona (bloqueante) — por isso é chamada via
    asyncio.to_thread, para não travar o loop de eventos do FastAPI
    enquanto o Whisper processa o áudio."""

    if SALVAR_AUDIO_DEBUG:
        nome_arquivo = os.path.join(PASTA_DEBUG, f"bloco_{time.time():.0f}.wav")
        with open(nome_arquivo, "wb") as f:
            f.write(audio_bytes)
        print(f"[debug] áudio salvo em: {nome_arquivo}")

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
    mongo_status = "não configurado"
    if transcript_collection is not None:
        try:
            await mongo_client.admin.command("ping")
            mongo_status = "conectado"
        except Exception as exc:  # noqa: BLE001
            mongo_status = f"erro: {exc}"

    return {"status": "ok", "modelo": MODEL_SIZE, "mongodb": mongo_status}


@app.post("/transcribe")
async def transcrever_audio_completo(audio: UploadFile = File(...)):
    """Recebe um único arquivo de áudio completo (gravação inteira, do
    início ao fim) e transcreve tudo de uma vez — sem tempo real, sem
    divisão em blocos. Mais simples e mais preciso, já que o Whisper
    recebe o contexto completo da fala de uma só vez."""
    audio_bytes = await audio.read()
    texto = await asyncio.to_thread(transcrever_audio_completo_sync, audio_bytes)
    return {"text": texto}


def transcrever_audio_completo_sync(audio_bytes: bytes) -> str:
    if SALVAR_AUDIO_DEBUG:
        nome_arquivo = os.path.join(PASTA_DEBUG, f"completo_{time.time():.0f}.webm")
        with open(nome_arquivo, "wb") as f:
            f.write(audio_bytes)
        print(f"[debug] áudio completo salvo em: {nome_arquivo}")

    audio_buffer = io.BytesIO(audio_bytes)
    segments, info = model.transcribe(
        audio_buffer,
        language="pt",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    segments = list(segments)
    texto = " ".join(seg.text.strip() for seg in segments).strip()

    print(
        f"[debug] TRANSCRIÇÃO COMPLETA: "
        f"duracao_audio={info.duration:.2f}s "
        f"idioma_detectado={info.language} "
        f"n_segmentos={len(segments)} "
        f"tamanho_texto={len(texto)} caracteres"
    )

    return texto


# --- Curadoria / revisão do texto via LLM (BYOK — a chave é do usuário) ---

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"  # ver outras opções em console.groq.com/docs/models

PROMPT_REVISAO = """Você está revisando a transcrição de uma entrevista jurídica \
entre um advogado e seu cliente, gerada por um sistema de reconhecimento de voz.

Sua única tarefa é corrigir erros PROVÁVEIS DE TRANSCRIÇÃO — palavras que soam \
parecido foneticamente com o que provavelmente foi dito, mas que não fazem \
sentido no contexto (ex: "tese de dano moral" transcrito errado como "tese \
de dano mural").

Regras estritas:
- NÃO resuma, NÃO reescreva o estilo, NÃO corrija gramática coloquial normal \
de fala.
- NÃO adicione informação nenhuma que não esteja no texto original.
- NÃO remova nada, mesmo que pareça irrelevante.
- Se não tiver certeza se algo é erro de transcrição, MANTENHA como está \
(prefira não mexer a inventar uma correção errada).
- Responda APENAS com o texto corrigido, sem comentários, sem explicações, \
sem aspas ao redor.

Texto a revisar:
"""


class RevisarTextoRequest(BaseModel):
    texto: str
    groq_api_key: str = ""  # se vazio, cai no fallback do .env (só para testes)


@app.post("/revisar-transcricao")
async def revisar_transcricao(payload: RevisarTextoRequest):
    """Manda o texto já transcrito pela Whisper para uma LLM (Groq, via a
    chave do próprio usuário) corrigir prováveis erros fonéticos de
    transcrição, usando o contexto da frase para decidir."""

    if not payload.texto.strip():
        return {"texto_revisado": payload.texto}

    chave = payload.groq_api_key.strip() or os.getenv("GROQ_API_KEY", "")
    if not chave:
        return {
            "erro": (
                "Nenhuma chave da Groq informada (nem no campo da página, "
                "nem em GROQ_API_KEY no .env do backend)."
            )
        }

    headers = {
        "Authorization": f"Bearer {chave}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT_REVISAO},
            {"role": "user", "content": payload.texto},
        ],
        "temperature": 0.0,  # queremos correção conservadora, não criativa
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resposta = await client.post(GROQ_API_URL, headers=headers, json=body)

    if resposta.status_code != 200:
        return {
            "erro": f"Groq retornou status {resposta.status_code}: {resposta.text}"
        }

    dados = resposta.json()
    texto_revisado = dados["choices"][0]["message"]["content"].strip()

    print(
        f"[debug] revisão via Groq — "
        f"tamanho_original={len(payload.texto)} "
        f"tamanho_revisado={len(texto_revisado)}"
    )

    return {"texto_revisado": texto_revisado}


@app.post("/salvar-transcricao")
async def salvar_transcricao(payload: dict):
    """Salva o texto final (já revisado, quando possível) como um
    documento na collection 'transcript' do MongoDB Atlas (Sophia-Base
    → sophia-data → transcript). Mantém também uma cópia em .txt local
    como backup simples enquanto validamos o fluxo."""
    texto = (payload.get("texto") or "").strip()
    if not texto:
        return {"erro": "Texto vazio, nada para salvar."}

    agora = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Backup local em .txt (mantido por enquanto, redundante com o Mongo)
    os.makedirs("transcricoes", exist_ok=True)
    nome_arquivo_txt = f"transcricoes/transcricao_{agora}.txt"
    with open(nome_arquivo_txt, "w", encoding="utf-8") as f:
        f.write(texto)

    resultado = {"arquivo_txt": nome_arquivo_txt}

    if transcript_collection is not None:
        try:
            documento = {
                "text": texto,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            insercao = await transcript_collection.insert_one(documento)
            resultado["mongo_id"] = str(insercao.inserted_id)
            print(f"[debug] transcrição salva no MongoDB — id={insercao.inserted_id}")
        except Exception as exc:  # noqa: BLE001
            resultado["erro_mongo"] = str(exc)
            print(f"[debug] ERRO ao salvar no MongoDB: {exc}")
    else:
        resultado["aviso"] = "MONGODB_URI não configurada — salvo só localmente em .txt."

    return resultado


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