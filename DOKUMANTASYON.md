# asker_sms — Proje Dokümantasyonu

SMS üzerinden gelen mesajları bir LLM'e (OpenRouter) ileten, konuşma hafızasını
PostgreSQL'de kalıcı olarak tutan bir FastAPI + LangGraph uygulaması.

Bu doküman iki bölümden oluşur:

1. **Kod açıklaması** — her dosyanın ne yaptığı, satır satır.
2. **Veritabanı yönetimi** — kurulum, kontrol, temizleme ve canlıya çıkış.

---

## 1. Genel Mimari

İstek geldiğinde akış şöyledir:

```
HTTP isteği (curl / ileride NetGSM webhook)
        │
        ▼
app/api/chat.py        ──►  graph.invoke(...) + thread_id
        │
        ▼
LangGraph pipeline (app/agent/graph.py)
        │
        ├─ ingest_user_message        (kullanıcı metnini state'e ekler)
        ├─ build_openrouter_messages  (system prompt + geçmişi LLM formatına çevirir)
        └─ generate_response          (OpenRouter'a istek atar, cevabı ekler)
        │
        ▼
PostgresSaver (checkpointer)  ──►  PostgreSQL'e konuşma durumu yazılır
        │
        ▼
HTTP yanıtı: {"reply": "..."}
```

Temel fikirler:

- **Her adım bir "node"dur.** Node'lar sadece `state` okur ve değiştirecekleri
  alanları içeren bir sözlük (partial update) döndürür.
- **Hafıza `thread_id` ile izlenir.** Aynı `thread_id` (örn. telefon numarası) ile
  gelen istekler aynı konuşmayı sürdürür. Hafıza PostgreSQL'de tutulur.
- **Config tek yerden okunur.** Tüm gizli/ortam değerleri `app/core/config.py`
  üzerinden `.env`'den gelir.

Klasör yapısı:

```
app/
├── main.py              # FastAPI uygulaması + lifespan (DB bağlantısı)
├── core/
│   └── config.py        # Ortam değişkenleri / ayarlar
├── agent/
│   ├── state.py         # LangGraph state şeması
│   ├── prompts.py       # System prompt
│   ├── nodes.py         # Graph adımları (iş mantığı)
│   └── graph.py         # Pipeline tanımı + compile
└── api/
    ├── health.py        # /health, /ready
    └── chat.py          # /chat endpoint'i
```

---

## 2. Kod Açıklaması

### 2.1. `app/core/config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "FastAPI"
    app_env: str = "development"

    openrouter_api_key: str
    openrouter_model: str

    postgres_url: str

settings = Settings()
```

- `BaseSettings`: Pydantic'in ortam değişkeni / `.env` okuyan temel sınıfı.
- `model_config = SettingsConfigDict(env_file=".env", ...)`: Uygulama açılırken
  `.env` dosyasını otomatik okur, UTF-8 ile çözer.
- `app_name`, `app_env`: Varsayılan değerleri olan alanlar; `.env`'de yoksa
  varsayılan kullanılır.
- `openrouter_api_key`, `openrouter_model`, `postgres_url`: **Varsayılanı yok**,
  yani zorunlu. `.env`'de eksiklerse uygulama açılırken `ValidationError` ile
  durur. Bu bilinçli bir tercihtir: eksik konfigürasyon runtime'da değil, en
  başta fark edilir.
- `settings = Settings()`: Modül yüklenirken tek bir global ayar nesnesi üretir.
  Kodun geri kalanı `settings.postgres_url` gibi erişir.

> Not: `.env` içindeki değerlerde `=` işaretinden sonra boşluk bırakma alışkanlığı
> sorun çıkarabilir; `python-dotenv` baştaki boşluğu kırpsa da temiz yazmak en
> güvenlisidir: `POSTGRES_URL=postgresql://...`

### 2.2. `app/agent/state.py`

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    user_message: str
    agent_message: str
    messages: Annotated[list[BaseMessage], add_messages]
    openrouter_response: list[dict[str, str]]
```

`AgentState`, graph boyunca taşınan verinin şemasıdır.

- `user_message`: API'den gelen ham kullanıcı metni. İlk node bunu okur.
- `agent_message`: Üretilen son asistan cevabı. Endpoint bunu yanıt olarak döndürür.
- `messages`: Konuşma geçmişi (hafızanın kalbi). Tipi `Annotated[..., add_messages]`'tır:
  - `add_messages` bir **reducer**'dır. Bir node `messages` döndürdüğünde,
    LangGraph bunu mevcut listenin **üzerine yazmaz**, **sonuna ekler**.
  - Bu sayede her adım yeni mesaj eklerken geçmiş korunur.
- `openrouter_response`: OpenRouter'a gönderilecek hazır mesaj listesi
  (`[{"role": ..., "content": ...}, ...]`). "Hazırlama" ve "gönderme" node'larını
  ayırmak için ara bir alan olarak tutulur.

> İsimlendirme notu: Bu alan aslında "OpenRouter'a giden istek mesajları"nı tutuyor;
> `openrouter_request_messages` gibi bir ad daha açıklayıcı olurdu. Çalışmayı
> etkilemez ama ileride yeniden adlandırmak istersen hem `state.py` hem `nodes.py`
> birlikte güncellenmelidir.

### 2.3. `app/agent/prompts.py`

```python
SYSTEM_PROMPT = """
You are a concise and helpful SMS assistant.
Reply in short, clear Turkish sentences unless user asks otherwise.
""".strip()
```

- Modelin davranış kuralını (system prompt) tek yerde tutar.
- `.strip()` baş/sondaki boş satırları temizler.
- Prompt'u koddan ayırmak, güncellemeyi kolaylaştırır ve node kodunu sade tutar.

### 2.4. `app/agent/nodes.py`

Dosyanın başındaki sabitler:

```python
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = 30

ROLE_BY_MESSAGE_TYPE = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
}
```

- `OPENROUTER_URL`, `REQUEST_TIMEOUT`: Modül seviyesinde sabit; her çağrıda
  yeniden oluşturulmaz, "sihirli değer" olmaktan çıkar.
- `ROLE_BY_MESSAGE_TYPE`: LangChain mesaj tiplerini (`system`/`human`/`ai`)
  OpenRouter'ın beklediği rollere (`system`/`user`/`assistant`) çevirir.

#### Node 1 — `ingest_user_message`

```python
def ingest_user_message(state: AgentState) -> AgentState:
    """Store the incoming user text in the conversation history."""
    user_text = state.get("user_message", "").strip()
    if not user_text:
        return {}
    return {"messages": [HumanMessage(content=user_text)]}
```

- `state.get("user_message", "")`: Ham kullanıcı metnini güvenli şekilde okur.
- `if not user_text: return {}`: Boş mesajda state'i değiştirmez (gereksiz LLM
  çağrısını engeller). Boş sözlük "değişiklik yok" demektir.
- `return {"messages": [HumanMessage(...)]}`: Kullanıcı mesajını geçmişe eklenecek
  formatta döndürür. `add_messages` reducer'ı bunu geçmişe ekler.

#### Node 2 — `build_openrouter_messages`

```python
def build_openrouter_messages(state: AgentState) -> AgentState:
    """Prepend the system prompt and convert history to OpenRouter payload."""
    conversation: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state.get("messages", []),
    ]
    payload = [
        {"role": ROLE_BY_MESSAGE_TYPE[message.type], "content": message.content}
        for message in conversation
        if message.type in ROLE_BY_MESSAGE_TYPE
    ]
    return {"openrouter_response": payload}
```

- `conversation`: System prompt'u en başa koyar, ardından geçmiş mesajları ekler.
  Böylece model her çağrıda hem kuralı hem geçmişi görür.
- `payload`: LangChain mesaj nesnelerini OpenRouter'ın düz JSON formatına çevirir.
  Tanınmayan mesaj tipleri (`if message.type in ...`) atlanır — savunmacı kod.
- `return {"openrouter_response": payload}`: Hazır payload'ı state'e koyar.

#### Node 3 — `generate_response`

```python
def generate_response(state: AgentState) -> AgentState:
    """Call OpenRouter and append the assistant reply to the conversation."""
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": settings.openrouter_model,
        "messages": state.get("openrouter_response", []),
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False,
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    assistant_text = response.json()["choices"][0]["message"]["content"].strip()
    return {
        "messages": [AIMessage(content=assistant_text)],
        "agent_message": assistant_text,
    }
```

- `headers`: OpenRouter kimlik doğrulaması (`Bearer <api_key>`) ve JSON içerik tipi.
- `data`: İstek gövdesi.
  - `model`: `.env`'den gelen model adı.
  - `messages`: Bir önceki node'un hazırladığı payload.
  - `temperature`: Yaratıcılık (0.7 dengeli).
  - `max_tokens`: Cevap uzunluğu üst sınırı (SMS için 500 makul).
  - `stream`: `False` → cevabı tek parça al.
- `requests.post(..., timeout=REQUEST_TIMEOUT)`: Zaman aşımı ile çağrı (asılı kalmayı
  engeller).
- `response.raise_for_status()`: HTTP hatası varsa exception fırlatır (sessiz hatayı
  önler).
- `assistant_text`: Yanıt JSON'undan asistan metnini çıkarır ve kırpar.
- `return {...}`: Cevabı hem hafızaya (`messages` → `AIMessage`) hem de düz
  `agent_message` alanına yazar.

### 2.5. `app/agent/graph.py`

```python
from langgraph.graph import END, START, StateGraph
from .nodes import build_openrouter_messages, generate_response, ingest_user_message
from .state import AgentState

def build_graph(checkpointer):
    """Build the SMS agent pipeline: ingest -> build payload -> generate."""
    graph = StateGraph(AgentState)

    graph.add_node("ingest_user_message", ingest_user_message)
    graph.add_node("build_openrouter_messages", build_openrouter_messages)
    graph.add_node("generate_response", generate_response)

    graph.add_edge(START, "ingest_user_message")
    graph.add_edge("ingest_user_message", "build_openrouter_messages")
    graph.add_edge("build_openrouter_messages", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile(checkpointer=checkpointer)
```

- `StateGraph(AgentState)`: State şemasıyla bir graph oluşturur.
- `add_node(...)`: Üç iş adımını isimleriyle kaydeder.
- `add_edge(...)`: Adımların sırasını belirler:
  `START → ingest → build → generate → END`.
- `build_graph(checkpointer)`: Checkpointer'ı **dışarıdan parametre** olarak alır.
  Böylece graph, hafızanın nerede tutulduğundan (Postgres/memory) bağımsız kalır;
  test ve esneklik artar.
- `graph.compile(checkpointer=checkpointer)`: Graph'i çalıştırılabilir hale getirir.
  **Checkpointer verildiği için** her adımda state PostgreSQL'e kaydedilir —
  kalıcı hafızanın merkez noktası burasıdır.

### 2.6. `app/api/chat.py`

```python
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    user_id: str  # SMS senaryosunda telefon numarası burada kullanılır
    message: str

class ChatResponse(BaseModel):
    reply: str

@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    graph = request.app.state.agent_graph

    result = graph.invoke(
        {"user_message": payload.message},
        config={"configurable": {"thread_id": payload.user_id}},
    )
    return ChatResponse(reply=result["agent_message"])
```

- Bu dosya **HTTP katmanı**dır; iş mantığı içermez, sadece giriş/çıkış yapar.
- `ChatRequest` / `ChatResponse`: Pydantic ile tip güvenli istek/yanıt şeması.
- `request.app.state.agent_graph`: Uygulama açılışında (lifespan) bir kez derlenip
  saklanan graph'i alır.
- `graph.invoke({"user_message": ...}, config={...})`:
  - İlk argüman başlangıç state'i: kullanıcı mesajı.
  - `thread_id = payload.user_id`: **Hafızanın anahtarı.** Aynı `user_id` ile gelen
    istekler aynı konuşmayı sürdürür; farklı `user_id`'ler birbirinden izoledir.
- `result["agent_message"]`: Graph çıktısından son cevabı alıp döndürür.

### 2.7. `app/api/health.py`

```python
router = APIRouter(prefix="", tags=["health"])

@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

@router.get("/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready"}
```

- Basit sağlık kontrol uçları.
- `/health`: Süreç ayakta mı? (liveness)
- `/ready`: Trafik almaya hazır mı? (readiness)
- Railway/yük dengeleyici gibi ortamlarda izleme için kullanışlıdır.

### 2.8. `app/main.py`

```python
from contextlib import ExitStack, asynccontextmanager
from fastapi import FastAPI
from langgraph.checkpoint.postgres import PostgresSaver

from app.agent.graph import build_graph
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    stack = ExitStack()
    app.state.exit_stack = stack

    checkpointer = stack.enter_context(
        PostgresSaver.from_conn_string(settings.postgres_url)
    )
    checkpointer.setup()  # ilk kurulumda tablo/migration oluşturur

    app.state.agent_graph = build_graph(checkpointer)

    try:
        yield
    finally:
        stack.close()

app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.include_router(health_router)
app.include_router(chat_router)
```

- `lifespan`: Uygulamanın açılış/kapanış yaşam döngüsünü yönetir.
- `ExitStack`: Açılan kaynakların (DB bağlantısı) kapanışta düzgün temizlenmesini
  sağlar.
- `PostgresSaver.from_conn_string(settings.postgres_url)`: Connection string'den
  bir Postgres checkpointer açar. Bu bir context manager olduğu için `enter_context`
  ile stack'e bağlanır; kapanışta otomatik kapanır.
- `checkpointer.setup()`: **İlk kullanımda** gerekli tabloları/migration'ları
  oluşturur. İdempotent'tir; tekrar çağrılması zarar vermez.
- `build_graph(checkpointer)`: Graph'i checkpointer ile bir kez derler ve
  `app.state.agent_graph`'e koyar. Endpoint'ler buradan kullanır.
- `yield`: Uygulama bu noktada çalışır; kapanışta `finally` ile kaynaklar kapanır.
- `app.include_router(...)`: Health ve chat uçlarını uygulamaya bağlar.

> Neden bağlantı request içinde değil, lifespan'de açılıyor? Çünkü DB bağlantısı
> pahalı ve uzun ömürlüdür. Her istekte aç/kapa yapmak hem yavaş hem hatalıdır;
> bir kez açıp tüm istekler boyunca paylaşmak doğrusudur.

---

## 3. Çalıştırma (Lokal)

### 3.1. Ortam değişkenleri (`.env`)

Proje kökünde `.env` dosyası şu üç değeri içermelidir:

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=deepseek/deepseek-v4-flash
POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/asker_sms
```

> `.env` asla git'e veya canlıya gönderilmez; `.gitignore` içinde olmalıdır.
> Canlıda (Railway) değerler panel üzerinden "Variables" olarak girilir.

### 3.2. Bağımlılıklar

Gerekli paketler `requirement.txt` içindedir. Postgres sürücüsü için
`psycopg[binary]` kurulu olmalıdır (libpq gömülü gelir):

```bash
source .venv/bin/activate
pip install "psycopg[binary]"
```

> Not: Build sistemlerinin çoğu `requirements.txt` (çoğul) arar. Canlıya çıkmadan
> önce dosya adını `requirement.txt` → `requirements.txt` yapman gerekebilir.

### 3.3. Uygulamayı başlat

```bash
uvicorn app.main:app --reload
```

`Application startup complete.` görmen, lifespan'in DB'ye bağlanıp tabloları
kurduğu anlamına gelir.

### 3.4. Uçtan uca test (curl)

```bash
# 1) İlk mesaj
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "+905551112233", "message": "Merhaba, benim adım Berke"}'

# 2) Aynı kullanıcı — hafıza testi (modelin "Berke"yi hatırlaması beklenir)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "+905551112233", "message": "Benim adım neydi?"}'

# 3) Farklı kullanıcı — izolasyon testi (bilmemeli)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "+905559998877", "message": "Benim adım neydi?"}'
```

> NetGSM (SMS sağlayıcısı) henüz bağlı olmadan da bu testler çalışır. `curl`,
> ileride NetGSM webhook'unun göndereceği isteği taklit eder. NetGSM yalnızca
> gerçek telefon ↔ uygulama uçlarını bağlamak için gerekir; çekirdek mantık
> (prompt + hafıza + LLM) ondan bağımsızdır.

---

## 4. Veritabanı Yönetimi

Hafıza, LangGraph'ın PostgreSQL'de oluşturduğu checkpoint tablolarında tutulur.
Genelde şu tablolar bulunur: `checkpoints`, `checkpoint_blobs`,
`checkpoint_writes`, `checkpoint_migrations`.

### 4.1. Docker ile Postgres kurmak

Docker kullanılıyorsa (Dockerfile gerekmez, hazır imaj çalıştırılır):

```bash
docker run -d --name asker_sms_pg \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=asker_sms \
  -p 5432:5432 \
  postgres:16
```

Kalıcı veri (container silinse bile veri dursun) istersen named volume ekle:

```bash
docker run -d --name asker_sms_pg \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=asker_sms -p 5432:5432 \
  -v asker_sms_pgdata:/var/lib/postgresql/data \
  postgres:16
```

### 4.2. Container durumu ve yaşam döngüsü

```bash
# Durum kontrolü
docker ps -a --filter "name=asker_sms_pg"

# Başlat / durdur (veriyi korur)
docker start asker_sms_pg
docker stop asker_sms_pg

# Container'ı sil (volume yoksa veri de gider)
docker rm -f asker_sms_pg

# İmajı da sil (disk boşalt)
docker rmi postgres:16
```

### 4.3. Bağlantı ve içeriği kontrol

```bash
# Postgres hazır mı?
docker exec asker_sms_pg pg_isready -U postgres

# Veritabanlarını listele
docker exec -it asker_sms_pg psql -U postgres -c "\l"

# Tabloları listele
docker exec -it asker_sms_pg psql -U postgres -d asker_sms -c "\dt"

# Kayıt sayısı
docker exec -it asker_sms_pg psql -U postgres -d asker_sms \
  -c "SELECT count(*) FROM checkpoints;"
```

### 4.4. Hafızayı (checkpoint) temizleme

**Seçenek A — Tüm hafızayı temizle (şema kalır, veri gider):**

```bash
docker exec -it asker_sms_pg psql -U postgres -d asker_sms \
  -c "TRUNCATE checkpoints, checkpoint_blobs, checkpoint_writes;"
```

> `checkpoint_migrations` tablosuna dokunma; o sadece şema versiyonunu tutar.

**Seçenek B — Tek bir kullanıcıyı (thread) temizle:**

```bash
docker exec -it asker_sms_pg psql -U postgres -d asker_sms -c \
  "DELETE FROM checkpoint_writes WHERE thread_id='+905551112233';
   DELETE FROM checkpoint_blobs  WHERE thread_id='+905551112233';
   DELETE FROM checkpoints       WHERE thread_id='+905551112233';"
```

**Seçenek C — Komple sıfırla (DB'yi düşür ve yeniden oluştur):**

```bash
docker exec -it asker_sms_pg psql -U postgres -c "DROP DATABASE asker_sms;"
docker exec -it asker_sms_pg psql -U postgres -c "CREATE DATABASE asker_sms;"
```

> Sonra `uvicorn` yeniden başlatıldığında `checkpointer.setup()` tabloları yeniden
> oluşturur.

### 4.5. Yaygın hatalar ve çözümleri

| Hata | Sebep | Çözüm |
|------|-------|-------|
| `ImportError: no pq wrapper available` | `psycopg` libpq bulamıyor | `pip install "psycopg[binary]"` |
| `ValidationError: postgres_url Field required` | `.env`'de `POSTGRES_URL` yok | `.env`'e `POSTGRES_URL=...` ekle |
| `connection refused ... port 5432` | Postgres ayakta değil | Container'ı başlat (`docker start` / `docker run`) |
| OpenRouter 401/403 | API key/model hatalı | `.env`'deki `OPENROUTER_*` değerlerini kontrol et |

---

## 5. Canlıya Çıkış (Railway) — Sonraki Faz

Kod yapısı zaten Railway'e uygundur (her şey `POSTGRES_URL` üzerinden config'ten
gelir). Yapılması gerekenler:

1. **`requirement.txt` → `requirements.txt`** (build sistemleri çoğul ad arar).
2. Railway'de **PostgreSQL servisi** ekle (managed; Docker gerekmez).
3. Uygulama servisine `POSTGRES_URL` değişkenini bağla. Railway genelde
   `DATABASE_URL` verir; uygulama değişkenine `${{Postgres.DATABASE_URL}}` referansı
   verebilir veya config alanını `database_url` olarak adlandırabilirsin.
4. **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   (canlıda `--reload` kullanma).
5. Gerekirse bağlantı string'ine `?sslmode=require` ekle.
6. `checkpointer.setup()` her açılışta idempotent çalıştığı için migration derdi
   olmadan tablolar hazır olur.

Uygulama için **Dockerfile zorunlu değildir**; Railway Nixpacks ile Python projesini
otomatik algılar. Daha fazla kontrol istenirse Dockerfile opsiyonel olarak eklenebilir.

---

## 6. NetGSM Entegrasyonu — İleri Aşama

Mevcut `/chat` ucu, `thread_id = user_id` mantığıyla tasarlandığı için SMS'e
geçişte hafıza mimarisi **değişmez**. Eklenecekler:

- **Inbound (gelen SMS):** NetGSM'in webhook'undan `from_phone` ve `text` alınır;
  `graph.invoke({"user_message": text}, config={"configurable": {"thread_id": from_phone}})`
  çağrılır. (Örn. yeni bir `/sms/inbound` ucu.)
- **Outbound (giden SMS):** Graph'in ürettiği cevap, NetGSM API'sine gönderilerek
  kullanıcıya SMS olarak iletilir.

Yani NetGSM yalnızca uçları bağlar; ortadaki LLM + hafıza akışı hazırdır.
