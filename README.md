# Sophia — Protótipo de Transcrição (100% local)

Dois serviços separados:

- **backend/** — API em Python (FastAPI) que recebe blocos de áudio via
  WebSocket e transcreve **localmente** usando `faster-whisper` (rodando
  na CPU). Nenhum áudio sai da sua máquina — sem SaaS externo.
- **frontend/** — página HTML/JS simples que grava o microfone em blocos
  de ~4 segundos e mostra o texto transcrito conforme cada bloco volta.

## 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Rodar o servidor:

```bash
uvicorn main:app --reload --port 8000
```

Na **primeira vez** que você rodar, o `faster-whisper` vai baixar o
modelo `small` automaticamente (precisa de internet só nessa hora — fica
em cache local depois, e o resto funciona 100% offline).

Teste rápido: abra `http://localhost:8000/health` — deve retornar
`{"status": "ok", "modelo": "small"}`.

## 2. Frontend

Não precisa de build. Só abra `frontend/index.html` direto no navegador
(ou sirva com `python -m http.server 5500` dentro da pasta `frontend/`).

Clique em **"Iniciar gravação"**, permita o microfone, e fale — a cada
~4 segundos um bloco de áudio é transcrito e o texto vai sendo
completado na tela.

## Sobre a escolha do modelo

O código usa o modelo `small` do Whisper, rodando em CPU com
quantização `int8` (bom equilíbrio entre velocidade e precisão para
uso local sem GPU). Se estiver lento demais na sua máquina, edite
`backend/main.py` e troque `MODEL_SIZE` por `"base"` ou `"tiny"`
(mais rápidos, um pouco menos precisos). Se no futuro rodar em um
servidor com GPU, dá pra trocar para `device="cuda"` e usar modelos
maiores (`medium`, `large-v3`) com muito mais precisão.

## Por que blocos de 4 segundos, e não streaming contínuo?

O Whisper não foi feito para transcrever palavra-por-palavra em tempo
real como serviços de streaming (ex: Deepgram). Por isso, o frontend
grava pequenos blocos de áudio independentes (cada um um arquivo
válido) e manda um de cada vez. O backend transcreve cada bloco assim
que chega. Resultado: a transcrição aparece "quase ao vivo", com um
atraso de poucos segundos — mas 100% local e sem depender de nenhum
provedor externo.

## Observações importantes

- Este é um protótipo de validação da funcionalidade central. Para
  produção, ainda faltam: HTTPS/WSS, autenticação, persistência da
  transcrição por caso, e a camada de sanitização de documentos —
  tudo já mapeado no Documento de Contexto do Sophia.
- Rodar Whisper localmente custa CPU/tempo de processamento — em
  volume alto (muitos advogados usando ao mesmo tempo), isso pode
  virar um gargalo que uma API paga resolveria com mais facilidade.
  Vale reavaliar essa escolha conforme o produto crescer.
