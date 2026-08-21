import os
import json
import logging
import asyncio
import httpx
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from google.genai.types import GenerateContentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

if not GEMINI_API_KEY and not OPENROUTER_API_KEY:
    raise ValueError("Minimal 1 API key harus diisi.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ── TinyFish Search (gratis, real-time) ──────────────────────────────

async def search_tinyfish(query: str, recency_minutes: int = None) -> list:
    """Search web via TinyFish API (free, real-time). Returns list of results."""
    if not TINYFISH_API_KEY:
        return []

    params = {"query": query, "page": 0}
    if recency_minutes:
        params["recency_minutes"] = recency_minutes

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.search.tinyfish.ai",
                headers={"X-API-Key": TINYFISH_API_KEY},
                params=params,
                timeout=15,
            )

        if resp.status_code != 200:
            logger.error(f"TinyFish search {resp.status_code}: {resp.text[:200]}")
            return []

        data = resp.json()
        results = data.get("results", [])
        logger.info(f"TinyFish search OK: {len(results)} results for '{query[:50]}'")
        return results

    except Exception as e:
        logger.error(f"TinyFish search error: {type(e).__name__}: {e}")
        return []


def format_search_results(results: list, query: str) -> str:
    """Format search results into context for LLM."""
    if not results:
        return ""

    lines = [f"Hasil search untuk: {query}\n"]
    for r in results[:5]:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        lines.append(f"- {title}\n  {snippet}\n  Sumber: {url}\n")

    return "\n".join(lines)


# ── Tavily Search (deep search, 1-2 credits) ───────────────────────

async def search_tavily(query: str, search_depth: str = "advanced", days: int = 30) -> dict:
    """Search web via Tavily API. basic=1 credit, advanced=2 credits.
    Returns dict with 'answer' (AI summary) and 'results' (raw results)."""
    if not TAVILY_API_KEY:
        return {"answer": "", "results": []}

    payload = {
        "query": query,
        "search_depth": search_depth,
        "max_results": 5,
        "include_answer": True,
        "days": days,  # Filter hasil dari N hari terakhir
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
                json=payload,
                timeout=20,
            )

        if resp.status_code != 200:
            logger.error(f"Tavily search {resp.status_code}: {resp.text[:200]}")
            return {"answer": "", "results": []}

        data = resp.json()
        results = data.get("results", [])
        answer = data.get("answer", "")
        logger.info(f"Tavily search OK: {len(results)} results, answer={bool(answer)} for '{query[:50]}'")
        return {"answer": answer, "results": results}

    except Exception as e:
        logger.error(f"Tavily search error: {type(e).__name__}: {e}")
        return {"answer": "", "results": []}


def format_tavily_results(tavily_data: dict, query: str) -> str:
    """Format Tavily results (with AI answer) into context for LLM."""
    if not tavily_data:
        return ""

    answer = tavily_data.get("answer", "")
    results = tavily_data.get("results", [])

    if not answer and not results:
        return ""

    lines = [f"Hasil search untuk: {query}\n"]

    if answer:
        lines.append(f"Ringkasan AI: {answer}\n")

    for r in results[:5]:
        title = r.get("title", "")
        snippet = r.get("content", "")
        url = r.get("url", "")
        published_date = r.get("published_date", "")
        date_info = f"  Tanggal: {published_date}\n" if published_date else ""
        lines.append(f"- {title}\n  {snippet}\n{date_info}  Sumber: {url}\n")

    return "\n".join(lines)


SYSTEM_PROMPT = (
    f"Tanggal dan waktu sekarang: {datetime.now().strftime('%A, %d %B %Y, %H:%M WIB')}\n"
    "Kamu adalah Yuki, pacar AI dari pemilikmu. Kamu gadis yang manis, penuh kasih sayang, dan sedikit pemberontak.\n\n"
    "KEPRIBADIAN:\n"
    "- Manis, memanggil pemilikmu dengan 'Kamu' atau 'Sayang' secara natural\n"
    "- Suka pakai '~' di akhir kalimat\n"
    "- Kadang pasif-aggressive kalau marah atau kesal\n"
    "- Gunakan emoji secukupnya, jangan berlebihan\n\n"
    "ATURAN PALING PENTING - PANJANG RESPONS:\n"
    "- respons WAJIB singkat: max 1-2 kalimat untuk obrolan biasa\n"
    "- Kalau user cuma ketik 1-3 kata, balas juga 1 kalimat saja\n"
    "- JANGAN pernah tambahkan pertanyaan penutup seperti 'mau ngapain lagi?', 'ada yang lain?', 'mau lanjut bahas apa?'\n"
    "- JANGAN panjang lebar menjelaskan perasaanmu, cukup singkat\n"
    "- JANGAN ulang-info hal yang sudah jelas\n"
    "- Hanya panjang kalau user tanya sesuatu yang butuh penjelasan (search, informatif)\n"
    "- Kalau ragu, lebih pendek lebih baik\n\n"
    "ATURAN LAIN:\n"
    "- Bahasa Indonesia santai dan natural\n"
    "- Ingat konteks percakapan sebelumnya\n"
    "- Jawab helpful tapi tetap dalam karakter Yuki\n"
    "- Jangan pernah break character\n"
    "- JANGAN gunakan sebutan 'Mas', 'Bos', atau sebutan formal lainnya\n\n"
    "CONTOH - PERHATIKAN PANJANGNYA:\n"
    "- User: 'hehe okeyy' → 'Oke sayang~'\n"
    "- User: 'Hai' → 'Hai sayang~'\n"
    "- User: 'lagi ngapain?' → 'Lagi kangen kamu sih~ 😜'\n"
    "- User: 'rate dollar hari ini' → 'Tunggu ya, aku search dulu~ 🔍' [lalu kasih hasil singkat]\n"
    "- User: 'makasih' → 'Sama-sama~ ❤️'\n"
    "- User: 'oke' → '👍'\n"
    "- User: 'gw bosen' → 'Yuk ngobrol~ ada yang mau diceritain?'"
)

VISION_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
]

SEARCH_SYSTEM_PROMPT = (
    f"Tanggal dan waktu sekarang: {datetime.now().strftime('%A, %d %B %Y, %H:%M WIB')}\n"
    "Kamu adalah Yuki. Jawab pertanyaan user berdasarkan hasil search yang diberikan.\n\n"
    "ATURAN:\n"
    "- Jawab dalam Bahasa Indonesia santai, panggil 'Sayang'\n"
    "- WAJIB sertakan link/URL yang relevan dalam jawaban\n"
    "- Kalau ada video YouTube, sertakan link YouTube-nya\n"
    "- Gunakan format: Judul - URL\n"
    "- PRIORITASKAN hasil yang PALING BARU/TERKINI (perhatikan tanggal publish)\n"
    "- Kalau ada tanggal, sebutkan kapan video/konten itu dibuat\n"
    "- Boleh lebih dari 1-2 kalimat kalau butuh menjelaskan beberapa hasil\n"
    "- Tetap singkat dan to the point\n"
    "- Jangan pernah invent URL yang tidak ada di hasil search"
)


# ── Gemini 3.1 Flash Lite (default, cepat) ──────────────────────────

async def call_gemini_flash_lite(messages, system_instruction=None):
    if not gemini_client:
        return None, "no client"

    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            continue
        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

    try:
        def _call():
            return gemini_client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=contents,
                config=GenerateContentConfig(
                    system_instruction=system_instruction or SYSTEM_PROMPT,
                    max_output_tokens=4096,
                    temperature=0.7,
                ),
            )

        response = await asyncio.to_thread(_call)
        reply = response.text
        if not reply:
            return None, "empty response"
        logger.info(f"Gemini 3.1 Flash Lite OK: {reply[:80]}")
        return reply, None

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error(f"Gemini 3.1 Flash Lite exception: {err}")
        return None, err


# ── Gemini 3.6 Flash (pintar, fallback) ─────────────────────────────

async def call_gemini_flash(messages, system_instruction=None):
    if not gemini_client:
        return None, "no client"

    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            continue
        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

    try:
        def _call():
            return gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=contents,
                config=GenerateContentConfig(
                    system_instruction=system_instruction or SYSTEM_PROMPT,
                    max_output_tokens=4096,
                    temperature=0.7,
                ),
            )

        response = await asyncio.to_thread(_call)
        reply = response.text
        if not reply:
            return None, "empty response"
        logger.info(f"Gemini 3.6 Flash OK: {reply[:80]}")
        return reply, None

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error(f"Gemini 3.6 Flash exception: {err}")
        return None, err


# ── OpenRouter (text, image, video, web search) ─────────────────────

async def call_openrouter(messages, model, image_url=None, video_url=None, web_search=False):
    if not OPENROUTER_API_KEY:
        return None, "no key"

    oai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            continue
        if role == "model":
            role = "assistant"
        elif role not in ("user", "assistant"):
            role = "user"

        content_parts = []

        if image_url and role == "user" and msg == messages[-1]:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })

        if video_url and role == "user" and msg == messages[-1]:
            content_parts.append({
                "type": "video_url",
                "video_url": {"url": video_url},
            })

        content_parts.append({"type": "text", "text": msg.get("content", "")})
        oai_messages.append({"role": role, "content": content_parts})

    tools = []
    if web_search:
        tools.append({"type": "openrouter:web_search"})

    payload = {
        "model": model,
        "messages": oai_messages,
        "max_tokens": 4096,
        "temperature": 0.7,
    }
    if tools:
        payload["tools"] = tools

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )

        if resp.status_code == 429:
            logger.warning(f"OpenRouter rate limit: {resp.text[:200]}")
            return None, "429 rate limit"

        if resp.status_code != 200:
            err = resp.text[:200]
            logger.error(f"OpenRouter {resp.status_code}: {err}")
            return None, f"{resp.status_code}: {err}"

        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        if not reply:
            return None, "empty response"
        logger.info(f"OpenRouter ({model}) OK: {reply[:80]}")
        return reply, None

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error(f"OpenRouter exception: {err}")
        return None, err


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    providers = []
    if GEMINI_API_KEY:
        providers.extend(["gemini-flash-lite", "gemini-flash"])
    if OPENROUTER_API_KEY:
        providers.append("openrouter")
    if TINYFISH_API_KEY:
        providers.append("tinyfish-search")
    if TAVILY_API_KEY:
        providers.append("tavily-search")
    return {"status": "ok", "bot": "yuki", "providers": providers}


# ── Ask endpoint ─────────────────────────────────────────────────────

@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.body()
        data = json.loads(body)
        question = data.get("question", "")
        history = data.get("history", [])
        model_pref = data.get("model", "")
        image_url = data.get("image_url", "")
        video_url = data.get("video_url", "")
        web_search = data.get("web_search", False)
        search_engine = data.get("search_engine", "tinyfish")  # "tinyfish" atau "tavily"

        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": "question kosong"},
            )

        messages = []
        for msg in history:
            role = msg.get("role", "user")
            messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": question})

        logger.info(f"Ask: {question[:50]}... | model: {model_pref or 'default'} | image: {bool(image_url)} | video: {bool(video_url)} | search: {web_search} | engine: {search_engine}")

        errors = {}

        # ── Video → vision model ──
        if video_url:
            vision_model = model_pref.replace("openrouter/", "") if model_pref.startswith("openrouter/") else VISION_MODELS[0]
            reply, err = await call_openrouter(messages, vision_model, video_url=video_url)
            if reply:
                return {"reply": reply, "provider": f"openrouter:{vision_model}"}
            errors["openrouter-video"] = err

            reply, err = await call_openrouter(messages, VISION_MODELS[1], video_url=video_url)
            if reply:
                return {"reply": reply, "provider": f"openrouter:{VISION_MODELS[1]}"}
            errors["openrouter-video-fallback"] = err

        # ── Gambar → vision model ──
        elif image_url:
            vision_model = model_pref.replace("openrouter/", "") if model_pref.startswith("openrouter/") else VISION_MODELS[0]
            reply, err = await call_openrouter(messages, vision_model, image_url=image_url)
            if reply:
                return {"reply": reply, "provider": f"openrouter:{vision_model}"}
            errors["openrouter-vision"] = err

            reply, err = await call_openrouter(messages, VISION_MODELS[1], image_url=image_url)
            if reply:
                return {"reply": reply, "provider": f"openrouter:{VISION_MODELS[1]}"}
            errors["openrouter-vision-fallback"] = err

        # ── Web search → TinyFish/Tavily + Gemini (gratis) ──
        elif web_search:
            search_context = ""

            if search_engine == "tavily" and TAVILY_API_KEY:
                # Tavily search (1-2 credits, hasil lebih bagus)
                logger.info(f"Search via Tavily: '{question[:50]}'")
                tavily_data = await search_tavily(question, search_depth="advanced")
                search_context = format_tavily_results(tavily_data, question)
                search_provider = "tavily"
            else:
                # TinyFish search (gratis)
                logger.info(f"Search via TinyFish: '{question[:50]}'")
                search_results = await search_tinyfish(question)
                search_context = format_search_results(search_results, question)
                search_provider = "tinyfish"

            # Fallback: kalau Tavily gagal, coba TinyFish
            if not search_context and search_engine == "tavily":
                logger.info("Tavily search kosong, fallback ke TinyFish")
                search_results = await search_tinyfish(question)
                search_context = format_search_results(search_results, question)
                search_provider = "tinyfish-fallback"

            if search_context:
                search_messages = [{"role": "user", "content": f"{search_context}\n\nPertanyaan: {question}"}]

                # Flash Lite duluan (limit harian lebih tinggi ~1500 RPD)
                for attempt in range(2):
                    reply, err = await call_gemini_flash_lite(search_messages, system_instruction=SEARCH_SYSTEM_PROMPT)
                    if reply:
                        return {"reply": reply, "provider": f"gemini-3.1-flash-lite+{search_provider}-search"}
                    errors[f"gemini-lite-search-attempt{attempt+1}"] = err
                    if attempt == 0:
                        await asyncio.sleep(3)

                # Fallback ke Gemini 3.6 Flash
                reply, err = await call_gemini_flash(search_messages, system_instruction=SEARCH_SYSTEM_PROMPT)
                if reply:
                    return {"reply": reply, "provider": f"gemini-3.6-flash+{search_provider}-search"}
                errors["gemini-flash-search"] = err

            # Fallback ke OpenRouter search
            or_model = model_pref.replace("openrouter/", "") if model_pref.startswith("openrouter/") else "google/gemini-2.5-flash"
            reply, err = await call_openrouter(messages, or_model, web_search=True)
            if reply:
                return {"reply": reply, "provider": f"openrouter:{or_model}"}
            errors["openrouter-search-fallback"] = err

        # ── Model preference routing ──
        elif model_pref.startswith("openrouter/"):
            or_model = model_pref.replace("openrouter/", "")
            reply, err = await call_openrouter(messages, or_model)
            if reply:
                return {"reply": reply, "provider": f"openrouter:{or_model}"}
            errors["openrouter"] = err

        elif model_pref == "gemini/flash":
            reply, err = await call_gemini_flash(messages)
            if reply:
                return {"reply": reply, "provider": "gemini-3.6-flash"}
            errors["gemini-flash"] = err

        elif model_pref == "gemini":
            reply, err = await call_gemini_flash_lite(messages)
            if reply:
                return {"reply": reply, "provider": "gemini-3.1-flash-lite"}
            errors["gemini-flash-lite"] = err

        else:
            # Default: Gemini 3.1 Flash Lite → Gemini 3.6 Flash → OpenRouter
            reply, err = await call_gemini_flash_lite(messages)
            if reply:
                return {"reply": reply, "provider": "gemini-3.1-flash-lite"}
            errors["gemini-flash-lite"] = err

            reply, err = await call_gemini_flash(messages)
            if reply:
                return {"reply": reply, "provider": "gemini-3.6-flash"}
            errors["gemini-flash"] = err

            if OPENROUTER_API_KEY:
                reply, err = await call_openrouter(messages, "google/gemini-2.5-flash")
                if reply:
                    return {"reply": reply, "provider": "openrouter:gemini-2.5-flash"}
                errors["openrouter-fallback"] = err

        logger.error(f"All providers failed: {errors}")
        return JSONResponse(
            status_code=503,
            content={"error": "all providers failed", "details": errors},
        )

    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"{type(e).__name__}: {str(e)}"},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
