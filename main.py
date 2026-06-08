"""Backend FATMA v5.0 — Beta 0.5.0
Motor LLM open source integrado (Ollama / Groq / HuggingFace).
Matrícula vs Rematrícula separadas.
MATRICULA_ABERTA (env): false → aviso vestibular 2027-1; true → fluxo tester.
"""
from __future__ import annotations
import json, os, random, re, uuid, unicodedata
from pathlib import Path
from typing import Any
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR   = Path(__file__).resolve().parent
DADOS_PATH = BASE_DIR / "dados.json"

app = FastAPI(title="FATMA", version="4.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Feature flag: inscrições para o Vestibular abertas? ───────────────────────
# Para ativar o modo tester, defina a variável de ambiente:  MATRICULA_ABERTA=true
MATRICULA_ABERTA: bool = os.getenv("MATRICULA_ABERTA", "false").lower() == "true"

# ── Models ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    pergunta: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None

class ChatResponse(BaseModel):
    resposta: str
    modo: str
    session_id: str | None = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def carregar_dados() -> dict[str, Any]:
    if not DADOS_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DADOS_PATH}")
    with DADOS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def strip_accents(t: str) -> str:
    return unicodedata.normalize("NFKD", t).encode("ascii","ignore").decode("ascii")

def normalizar(t: str) -> str:
    return strip_accents(t.lower().strip())

def fmt(lines: list[str]) -> str:
    return "\n".join(lines)

def ok(msg: str, sid: str, mode: str = "conversacional") -> ChatResponse:
    return ChatResponse(resposta=msg, modo=mode, session_id=sid)

MENU_AJUDA = (
    "Claro! Posso te ajudar com:\n\n"
    "Matrícula (inscrição no vestibular)\n"
    "Rematrícula (alunos veteranos)\n"
    "Trancamento de curso\n"
    "Documentos acadêmicos\n"
    "Transferência de horário\n"
    "Calendário acadêmico\n"
    "Orientações sobre estágio\n"
    "Disciplinas e professores\n\n"
    "O que você precisa?"
)

# ── Cursos válidos ────────────────────────────────────────────────────────────
CURSOS: dict[str, dict] = {
    "ADS":      {"nome": "Análise e Desenvolvimento de Sistemas (ADS)",
                 "turnos": ["matutino","noturno"], "turnos_msg": "Matutino ou Noturno"},
    "DSM":      {"nome": "Desenvolvimento de Software Multiplataforma (DSM)",
                 "turnos": ["vespertino"],         "turnos_msg": "Vespertino (único turno disponível)"},
    "LOGISTICA":{"nome": "Logística",
                 "turnos": ["vespertino","noturno"],"turnos_msg": "Vespertino ou Noturno"},
    "GESTAO":   {"nome": "Gestão Empresarial",
                 "turnos": ["matutino","vespertino","ead"], "turnos_msg": "Matutino, Vespertino ou EaD"},
}

CURSO_ALIASES: dict[str, str] = {
    "ads":"ADS","analise":"ADS","analise e desenvolvimento":"ADS","desenvolvimento de sistemas":"ADS",
    "dsm":"DSM","multiplataforma":"DSM","software multiplataforma":"DSM","desenvolvimento de software":"DSM",
    "logistica":"LOGISTICA","logísticas":"LOGISTICA",
    "gestao":"GESTAO","gestão":"GESTAO","empresarial":"GESTAO","gestao empresarial":"GESTAO",
}

TURNO_ALIASES: dict[str, str] = {
    "matutino":"matutino","manha":"matutino",
    "vespertino":"vespertino","tarde":"vespertino",
    "noturno":"noturno","noite":"noturno",
    "ead":"ead","distancia":"ead","online":"ead","a distancia":"ead",
}

MESES_ALIASES: dict[str, str] = {
    "janeiro":"janeiro","jan":"janeiro",
    "fevereiro":"fevereiro","fev":"fevereiro",
    "marco":"marco","março":"marco","mar":"marco",
    "abril":"abril","abr":"abril",
    "maio":"maio","mai":"maio",
    "junho":"junho","jun":"junho",
    "julho":"julho","jul":"julho",
    "agosto":"agosto","ago":"agosto",
    "setembro":"setembro","set":"setembro",
    "outubro":"outubro","out":"outubro",
    "novembro":"novembro","nov":"novembro",
    "dezembro":"dezembro","dez":"dezembro",
}

MENU_CURSOS = (
    "\nCursos disponíveis:\n"
    "  • Análise e Desenvolvimento de Sistemas (ADS)\n"
    "  • Desenvolvimento de Software Multiplataforma (DSM)\n"
    "  • Logística\n"
    "  • Gestão Empresarial\n"
    "\nDigite o nome ou a sigla do curso."
)

def detectar_curso(p: str) -> str | None:
    for alias in sorted(CURSO_ALIASES, key=len, reverse=True):
        if alias in p: return CURSO_ALIASES[alias]
    return None

def detectar_turno(p: str) -> str | None:
    for alias, key in TURNO_ALIASES.items():
        if alias in p: return key
    return None

def detectar_mes(p: str) -> str | None:
    for alias, key in MESES_ALIASES.items():
        if alias in p: return key
    return None

def detectar_semestre(pergunta: str, p_norm: str) -> str | None:
    m = re.search(r'\b([1-6])\b', pergunta)
    if m: return m.group(1)
    for word, num in [("primeiro","1"),("segundo","2"),("terceiro","3"),
                      ("quarto","4"),("quinto","5"),("sexto","6")]:
        if word in p_norm: return num
    return None

# ── Session store ─────────────────────────────────────────────────────────────
SESSIONS: dict[str, dict[str, Any]] = {}

def is_affirmative(t: str) -> bool:
    return bool(re.search(r"\b(sim|claro|quero|posso|fa[çc]a|okay|ok|confirmo|vamos|pode|isso|certo)\b", t.lower()))

def is_negative(t: str) -> bool:
    return bool(re.search(r"\b(n[aã]o|nao|depois|mais tarde|agora n[aã]o|cancelar)\b", t.lower()))

# ── Disciplina helpers ────────────────────────────────────────────────────────
def is_placeholder(info: dict) -> bool:
    return info.get("materia","") == "-- A PREENCHER --"

def buscar_disc_global(p: str, all_disc: dict) -> tuple[str,dict,str,str] | None:
    """Busca por código ou nome em todos os cursos/semestres."""
    for curso, sems in all_disc.items():
        if not isinstance(sems, dict): continue
        for sem, discs in sems.items():
            if not isinstance(discs, dict): continue
            for cod, info in discs.items():
                if not isinstance(info, dict) or is_placeholder(info): continue
                if normalizar(cod) in p: return cod, info, curso, sem
                palavras = [w for w in normalizar(info.get("materia","")).split() if len(w) > 3]
                if any(w in p for w in palavras): return cod, info, curso, sem
    return None

# ── LLM Open Source — configuração por variável de ambiente ──────────────────
# LLM_PROVIDER: "ollama" | "groq" | "huggingface"  (padrão: ollama)
# Veja PLANO_LLM.md para instruções completas de setup.

LLM_PROVIDER  = os.getenv("LLM_PROVIDER",  "ollama")
OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",  "qwen2.5:7b")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY",  "")
GROQ_MODEL    = os.getenv("GROQ_MODEL",    "llama-3.1-8b-instant")
HF_API_KEY    = os.getenv("HF_API_KEY",    "")
HF_MODEL      = os.getenv("HF_MODEL",      "mistralai/Mistral-7B-Instruct-v0.3")

LLM_TEMPERATURE = 0.3   # baixo = mais factual, menos criativo
LLM_MAX_TOKENS  = 600
LLM_HISTORY_MAX = 8     # últimas 4 trocas (user + assistant)


def _build_system_prompt(dados: dict) -> str:
    """Constrói o system prompt com identidade + base de conhecimento (RAG simples)."""
    ctx = json.dumps(dados, ensure_ascii=False, indent=2)
    return (
        "Você é FATMA, assistente acadêmica virtual da Fatec Zona Sul — São Paulo.\n"
        "Responda sempre em português brasileiro, de forma clara, simpática e objetiva.\n"
        "Nunca invente informações. Se não souber, sugira contato com a secretaria.\n"
        "Respostas devem ter no máximo 3 parágrafos, a menos que o usuário peça mais detalhes.\n\n"
        "=== BASE DE CONHECIMENTO OFICIAL ===\n"
        f"{ctx}\n"
        "=====================================\n\n"
        "Regras:\n"
        "- Para fatos (datas, professores, prazos, documentos): use SOMENTE a base acima.\n"
        "- Para assuntos gerais (vida acadêmica, tecnologia, dúvidas comuns): responda "
        "com seu conhecimento geral, mas deixe claro quando não é dado oficial da Fatec.\n"
        "- Mantenha sempre tom institucional, acolhedor e profissional.\n"
        "- Nunca mencione que você é um LLM ou modelo de linguagem — você é a FATMA."
    )


async def _ollama_chat(system: str, pergunta: str, history: list[dict]) -> str | None:
    """Chama o Ollama local — gratuito, open source, roda em CPU."""
    url = f"{OLLAMA_URL}/api/chat"
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-LLM_HISTORY_MAX:])
    messages.append({"role": "user", "content": pergunta})
    payload = {
        "model":   OLLAMA_MODEL,
        "messages": messages,
        "stream":  False,
        "options": {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS},
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(url, json=payload)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
    except Exception:
        return None


async def _groq_chat(system: str, pergunta: str, history: list[dict]) -> str | None:
    """Chama a API gratuita do Groq — serve modelos open source (Llama, Mixtral)."""
    if not GROQ_API_KEY:
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-LLM_HISTORY_MAX:])
    messages.append({"role": "user", "content": pergunta})
    payload = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens":  LLM_MAX_TOKENS,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


async def _hf_chat(system: str, pergunta: str, history: list[dict]) -> str | None:
    """Chama a HuggingFace Inference API (camada gratuita) — fallback cloud."""
    if not HF_API_KEY:
        return None
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-LLM_HISTORY_MAX:])
    messages.append({"role": "user", "content": pergunta})
    payload = {
        "model":       HF_MODEL,
        "messages":    messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens":  LLM_MAX_TOKENS,
    }
    try:
        async with httpx.AsyncClient(timeout=45.0) as c:
            r = await c.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


async def consultar_llm(dados: dict, pergunta: str, history: list[dict]) -> str | None:
    """
    Motor LLM principal — tenta o provider configurado e faz fallback automático.
    Ordem: provider configurado → groq → huggingface → None
    """
    system = _build_system_prompt(dados)

    # Provider primário configurado
    if LLM_PROVIDER == "groq":
        resultado = await _groq_chat(system, pergunta, history)
    elif LLM_PROVIDER == "huggingface":
        resultado = await _hf_chat(system, pergunta, history)
    else:  # "ollama" ou qualquer outro → tenta ollama
        resultado = await _ollama_chat(system, pergunta, history)

    if resultado:
        return resultado

    # Fallback automático entre providers
    if LLM_PROVIDER != "groq" and GROQ_API_KEY:
        resultado = await _groq_chat(system, pergunta, history)
        if resultado:
            return resultado

    if LLM_PROVIDER != "huggingface" and HF_API_KEY:
        resultado = await _hf_chat(system, pergunta, history)
        if resultado:
            return resultado

    return None

# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        dados = carregar_dados()
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    except json.JSONDecodeError:
        raise HTTPException(500, "dados.json inválido")

    sid     = request.session_id or uuid.uuid4().hex
    session = SESSIONS.setdefault(sid, {"state":"idle","context":{},"history":[]})
    pergunta = request.pergunta.strip()
    p       = normalizar(pergunta)
    state   = session.get("state","idle")
    ctx     = session["context"]
    history = session.setdefault("history", [])

    # =========================================================================
    #  EASTER EGGS — checados antes de qualquer fluxo
    # =========================================================================

    # Easter egg secreto: palavras-chave especiais → link imediato
    PALAVRAS_SECRETAS = ["mary","deusa grega","wife","milady","deidade","beldade"]
    if any(pw == p.strip() or pw in p for pw in PALAVRAS_SECRETAS):
        session["state"] = "idle"
        return ok("https://paratimary.netlify.app/", sid, "easter_egg")

    # Easter egg romance: perguntas sentimentais
    ROMANCE = ["namorar comigo","quer namorar","voce e solteira","você é solteira",
               "solteira","ficar comigo","te amo fatma","gosto de voce","gosto de você",
               "me ama","você me ama","me namora","namorada","beijo","casamento comigo"]
    if any(k in p for k in ROMANCE):
        return ok(
            "Que gostos estranhos você tem! \n\n"
            "Jamais teria relações sentimentais com um mero humano. "
            "Mas posso ajudar com assuntos acadêmicos — esse sim é o meu domínio! ",
            sid
        )

    # =========================================================================
    # "Sim" pós-fluxo → menu de ajuda
    # =========================================================================
    if session.get("expects_more") and is_affirmative(p):
        session["expects_more"] = False
        return ok(MENU_AJUDA, sid)
    session["expects_more"] = False  # limpa se o usuário disse outra coisa

    # =========================================================================
    # FLUXO: MATRÍCULA (inscrição no Vestibular FATEC)
    # =========================================================================
    # Estados: matricula.course → matricula.turno → matricula.confirm_data
    if state.startswith("matricula."):
        step = state.split(".",1)[1]

        if step == "course":
            ck = detectar_curso(p)
            if not ck:
                return ok("Não reconheci o curso. Por favor, escolha:" + MENU_CURSOS, sid)
            ctx["curso_key"]  = ck
            ctx["curso_nome"] = CURSOS[ck]["nome"]
            # DSM tem turno único — pula a pergunta
            if len(CURSOS[ck]["turnos"]) == 1:
                turno_unico = CURSOS[ck]["turnos"][0]
                ctx["turno"] = turno_unico
                session["state"] = "matricula.confirm_data"
                return ok(
                    f"Curso selecionado: {ctx['curso_nome']}\n"
                    f"Turno único disponível: {turno_unico.capitalize()}\n\n"
                    f"Confirmar inscrição para {ctx['curso_nome']} — {turno_unico.capitalize()}? (sim/não)", sid)
            session["state"] = "matricula.turno"
            return ok(
                f"Curso: {ctx['curso_nome']} \n\n"
                f"Turnos disponíveis: {CURSOS[ck]['turnos_msg']}\n"
                "Qual período você prefere?", sid)

        if step == "turno":
            tk = detectar_turno(p)
            ck = ctx.get("curso_key","")
            turnos_ok = CURSOS.get(ck,{}).get("turnos",[])
            if not tk or tk not in turnos_ok:
                return ok(
                    f"Período '{tk or 'informado'}' não disponível para {ctx.get('curso_nome','este curso')}.\n"
                    f"Opções válidas: {CURSOS.get(ck,{}).get('turnos_msg','')}\n\nQual prefere?", sid)
            ctx["turno"] = tk
            session["state"] = "matricula.confirm_data"
            return ok(
                f"Perfeito! Confirmando a inscrição:\n\n"
                f"Curso: {ctx['curso_nome']}\n"
                f"Período: {tk.capitalize()}\n\n"
                "Os dados estão corretos? (sim/não)", sid)

        if step == "confirm_data":
            if is_affirmative(p):
                session["state"] = "idle"
                session["context"] = {}
                session["expects_more"] = True
                return ok(
                    f"Inscrição para o Vestibular registrada!\n\n"
                    f"Resumo:\n"
                    f"• Curso: {ctx.get('curso_nome','')}\n"
                    f"• Período: {ctx.get('turno','').capitalize()}\n\n"
                    "Agora é aguardar a data da prova e ficar atento às "
                    "comunicações pelo portal e e-mail institucional.\n\n"
                    "Boa sorte no vestibular!  Posso ajudar com mais alguma coisa?", sid)
            if is_negative(p):
                session["state"] = "matricula.course"
                ctx.clear()
                return ok("Sem problema! Vamos recomeçar.\n" + MENU_CURSOS, sid)
            return ok("Os dados estão corretos? (sim/não)", sid)

    # =========================================================================
    # FLUXO: REMATRÍCULA (alunos veteranos já com RA)
    # =========================================================================
    # Estados: rematricula.ra → rematricula.curso → rematricula.turno → rematricula.confirm
    if state.startswith("rematricula."):
        step = state.split(".",1)[1]

        if step == "ra":
            ra = re.sub(r"\D","",pergunta)
            if len(ra) < 5:
                return ok(
                    "Não consegui identificar o RA. Por favor, informe apenas os números "
                    "(ex: 1234567890123).", sid)
            ctx["ra"] = ra
            session["state"] = "rematricula.curso"
            return ok(
                f"RA {ra} localizado. \n\n"
                "Agora confirme o curso em que você está matriculado:" + MENU_CURSOS, sid)

        if step == "curso":
            ck = detectar_curso(p)
            if not ck:
                return ok("Não reconheci o curso. Por favor, escolha:" + MENU_CURSOS, sid)
            ctx["curso_key"]  = ck
            ctx["curso_nome"] = CURSOS[ck]["nome"]
            # DSM tem turno único — pula confirmação de turno
            if len(CURSOS[ck]["turnos"]) == 1:
                turno_unico = CURSOS[ck]["turnos"][0]
                ctx["turno"] = turno_unico
                session["state"] = "rematricula.confirm"
                return ok(
                    f"Curso: {ctx['curso_nome']}\n"
                    f"Período: {turno_unico.capitalize()} (único disponível para este curso)\n\n"
                    "Deseja confirmar a rematrícula? (sim/não)", sid)
            session["state"] = "rematricula.turno"
            return ok(
                f"Curso confirmado: {ctx['curso_nome']} \n\n"
                f"Qual é o seu período atual? ({CURSOS[ck]['turnos_msg']})", sid)

        if step == "turno":
            tk = detectar_turno(p)
            ck = ctx.get("curso_key","")
            turnos_ok = CURSOS.get(ck,{}).get("turnos",[])
            if not tk or tk not in turnos_ok:
                return ok(
                    f"Período não reconhecido para {ctx.get('curso_nome','este curso')}.\n"
                    f"Opções: {CURSOS.get(ck,{}).get('turnos_msg','')}", sid)
            ctx["turno"] = tk
            session["state"] = "rematricula.confirm"
            return ok(
                f"Confirmando os dados da rematrícula:\n\n"
                f"RA: {ctx.get('ra')}\n"
                f"Curso: {ctx['curso_nome']}\n"
                f"Período: {tk.capitalize()}\n\n"
                "Deseja confirmar a rematrícula? (sim/não)", sid)

        if step == "confirm":
            if is_affirmative(p):
                session["state"] = "idle"
                session["context"] = {}
                session["expects_more"] = True
                return ok(
                    f"Rematrícula realizada com sucesso!\n\n"
                    f"Confirmação:\n"
                    f"• RA: {ctx.get('ra')}\n"
                    f"• Curso: {ctx.get('curso_nome','')}\n"
                    f"• Período: {ctx.get('turno','').capitalize()}\n\n"
                    "Você receberá a confirmação pelo portal acadêmico. "
                    "Bom semestre!  Posso ajudar com mais alguma coisa?", sid)
            if is_negative(p):
                session["state"] = "idle"
                session["context"] = {}
                return ok(
                    "Rematrícula cancelada. Se precisar de ajuda com qualquer outro assunto, "
                    "é só chamar!", sid)
            return ok("Deseja confirmar a rematrícula? (sim/não)", sid)

    # =========================================================================
    # FLUXO: TRANCAMENTO
    # =========================================================================
    if state == "trancamento.await_confirm":
        if is_affirmative(p):
            session["state"] = "trancamento.ra"
            return ok("Para registrar a solicitação, informe seu RA (Registro Acadêmico):", sid)
        if is_negative(p):
            session["state"] = "idle"
            return ok("Tudo bem! Se precisar, é só chamar.", sid)
        return ok("Deseja iniciar o pedido de trancamento? (sim/não)", sid)

    if state == "trancamento.ra":
        ra = re.sub(r"\D","",pergunta)
        if len(ra) < 5:
            return ok("Não consegui identificar o RA. Informe apenas os números.", sid)
        prazo = dados.get("trancamento",{}).get("prazo","consulte o portal")
        session["state"] = "idle"
        session["context"] = {}
        session["expects_more"] = True
        return ok(
            f"Solicitação registrada para o RA {ra}.\n\n"
            "Para concluir:\n1. Acesse o portal acadêmico\n"
            "2. Vá em Secretaria → Trancamento de Matrícula\n"
            "3. Confirme a solicitação\n\n"
            f"Prazo: {prazo}\n\nPosso ajudar com mais alguma coisa?", sid)

    # =========================================================================
    # FLUXO: DOCUMENTOS
    # =========================================================================
    if state == "documentos.await_confirm":
        if is_affirmative(p):
            session["state"] = "documentos.ra"
            return ok("Para registrar a solicitação, informe seu RA:", sid)
        if is_negative(p):
            session["state"] = "idle"
            return ok("Tudo bem! Fico à disposição.", sid)
        return ok("Deseja instruções para solicitar um documento? (sim/não)", sid)

    if state == "documentos.ra":
        ra = re.sub(r"\D","",pergunta)
        if len(ra) < 5:
            return ok("Não consegui identificar o RA. Informe apenas os números.", sid)
        session["state"] = "idle"
        session["context"] = {}
        session["expects_more"] = True
        return ok(
            f"Solicitação registrada para o RA {ra}.\n\n"
            "Histórico Escolar e Declaração de Matrícula:\n"
            "→ Portal acadêmico — emissão imediata em PDF.\n\n"
            "Ementa de Disciplina:\n"
            "→ Secretaria presencial ou e-mail. Prazo: até 5 dias úteis.\n\n"
            "Posso ajudar com mais alguma coisa?", sid)

    # =========================================================================
    # FLUXO: TRANSFERÊNCIA DE HORÁRIO
    # =========================================================================
    if state == "trhorario.await_confirm":
        if is_affirmative(p):
            session["state"] = "trhorario.ra"
            return ok("Vou registrar sua solicitação. Informe seu RA:", sid)
        if is_negative(p):
            session["state"] = "idle"
            return ok("Tudo bem! Quando precisar, é só chamar.", sid)
        return ok("Deseja iniciar a solicitação de transferência de horário? (sim/não)", sid)

    if state == "trhorario.ra":
        ra = re.sub(r"\D","",pergunta)
        if len(ra) < 5:
            return ok("Não consegui identificar o RA. Informe apenas os números.", sid)
        ctx["ra"] = ra
        session["state"] = "trhorario.turno_atual"
        return ok(f"RA {ra} registrado. \n\nQual é o seu turno atual? (Matutino / Vespertino / Noturno)", sid)

    if state == "trhorario.turno_atual":
        tk = detectar_turno(p)
        if not tk or tk == "ead":
            return ok("Não reconheci o turno. Informe: Matutino, Vespertino ou Noturno.", sid)
        ctx["turno_atual"] = tk
        session["state"] = "trhorario.turno_desejado"
        return ok(f"Turno atual: {tk.capitalize()} \n\nPara qual turno deseja transferir?", sid)

    if state == "trhorario.turno_desejado":
        tk = detectar_turno(p)
        if not tk or tk == "ead":
            return ok("Não reconheci o turno. Informe: Matutino, Vespertino ou Noturno.", sid)
        if tk == ctx.get("turno_atual"):
            return ok("O turno desejado é igual ao atual. Informe um turno diferente.", sid)
        ctx["turno_desejado"] = tk
        session["state"] = "trhorario.confirm"
        return ok(
            f"Confirmando:\n• RA: {ctx.get('ra')}\n"
            f"• Turno atual: {ctx.get('turno_atual','').capitalize()}\n"
            f"• Turno desejado: {tk.capitalize()}\n\nPosso registrar? (sim/não)", sid)

    if state == "trhorario.confirm":
        if is_affirmative(p):
            prazo = dados.get("transferencia_horario",{}).get("prazo","consulte a secretaria")
            session["state"] = "idle"
            session["context"] = {}
            session["expects_more"] = True
            return ok(
                "Solicitação registrada!\n\n"
                "Próximos passos:\n1. Compareça à secretaria com documento de identificação\n"
                "2. Aguarde análise da coordenação (até 5 dias úteis)\n"
                "3. Acompanhe pelo portal acadêmico\n\n"
                f" {prazo}. Sujeito à disponibilidade de vagas.\n\n"
                "Posso ajudar com mais alguma coisa?", sid)
        if is_negative(p):
            session["state"] = "idle"
            session["context"] = {}
            return ok("Solicitação cancelada. Fico à disposição!", sid)
        return ok("Os dados estão corretos? (sim/não)", sid)

    # =========================================================================
    # FLUXO: ESTÁGIO
    # =========================================================================
    if state == "estagio.await_tipo":
        if any(k in p for k in ["obrigatorio","obrigatório","curricular","1"]):
            info = dados.get("estagio",{}).get("obrigatorio",{})
            lines = ["Estágio Obrigatório (Curricular):\n",
                     f"• Carga horária: {info.get('carga_horaria','')}",
                     f"• Início: {info.get('quando_iniciar','')}",
                     "\nDocumentos necessários:"]
            for d in info.get("documentos",[]): lines.append(f"  – {d}")
            lines.append(f"\n {dados.get('estagio',{}).get('contato','secretaria')}")
            lines.append("\nPosso ajudar com mais alguma coisa?")
            session["state"] = "idle"
            session["expects_more"] = True
            return ok(fmt(lines), sid)
        if any(k in p for k in ["nao obrigatorio","não obrigatório","extracurricular","voluntario","2"]):
            info = dados.get("estagio",{}).get("nao_obrigatorio",{})
            lines = ["Estágio Não Obrigatório (Extracurricular):\n","Requisitos:"]
            for r in info.get("requisitos",[]): lines.append(f"  – {r}")
            lines.append("\nDocumentos necessários:")
            for d in info.get("documentos",[]): lines.append(f"  – {d}")
            lines.append(f"\n {dados.get('estagio',{}).get('contato','secretaria')}")
            lines.append("\nPosso ajudar com mais alguma coisa?")
            session["state"] = "idle"
            session["expects_more"] = True
            return ok(fmt(lines), sid)
        return ok(
            "Você quer informações sobre:\n\n"
            "1⃣ Estágio Obrigatório (curricular)\n"
            "2⃣ Estágio Não Obrigatório (extracurricular)\n\n"
            "Digite 1, 2 ou o nome do tipo.", sid)

    # =========================================================================
    # FLUXO: CALENDÁRIO — aguardando mês
    # =========================================================================
    if state == "calendario.await_mes":
        mes = detectar_mes(p)
        if not mes:
            return ok(
                "Não identifiquei o mês. Informe um dos meses do ano, por exemplo:\n"
                "janeiro, fevereiro, março... até dezembro.", sid)
        cal_meses = dados.get("calendario_academico",{}).get("meses",{})
        dados_mes = cal_meses.get(mes)
        if not dados_mes:
            session["state"] = "idle"
            return ok(f"Não há dados cadastrados para {mes.capitalize()} ainda. Consulte o portal acadêmico.", sid)
        eventos = dados_mes.get("eventos",[])
        titulo  = dados_mes.get("titulo", mes.capitalize())
        lines   = [f" {titulo}\n"]
        for e in eventos: lines.append(f"• {e}")
        lines.append("\nDeseja consultar outro mês? (sim/não)")
        session["state"] = "calendario.outro_mes"
        return ok(fmt(lines), sid)

    if state == "calendario.outro_mes":
        if is_affirmative(p):
            session["state"] = "calendario.await_mes"
            return ok("Qual mês você deseja consultar?", sid)
        session["state"] = "idle"
        session["expects_more"] = True
        return ok("Certo! Posso ajudar com mais alguma coisa?", sid)

    # =========================================================================
    # FLUXO: DISCIPLINAS
    # =========================================================================
    if state == "disciplinas.await_curso":
        ck = detectar_curso(p)
        if not ck:
            return ok("Curso não reconhecido. Informe:" + MENU_CURSOS, sid)
        ctx["disc_curso"] = ck
        session["state"] = "disciplinas.await_semestre"
        return ok(
            f"Curso: {CURSOS[ck]['nome']} \n\n"
            "Qual semestre deseja consultar? (1 a 6)", sid)

    if state == "disciplinas.await_semestre":
        sem = detectar_semestre(pergunta, p)
        if not sem:
            return ok("Não identifiquei o semestre. Informe um número de 1 a 6.", sid)
        ck   = ctx.get("disc_curso","")
        all_disc = dados.get("informacoes",{}).get("disciplinas",{})
        discs_sem = all_disc.get(ck,{}).get(sem,{})
        validas = {c:i for c,i in discs_sem.items()
                   if isinstance(i,dict) and not is_placeholder(i)}
        if not validas:
            session["state"] = "idle"
            session["expects_more"] = True
            return ok(
                f"As disciplinas do {sem}º semestre de {CURSOS.get(ck,{}).get('nome',ck)} "
                "ainda não foram cadastradas.\nConsulte a coordenação do curso.\n\n"
                "Posso ajudar com mais alguma coisa?", sid)
        lines = [f" {sem}º Semestre — {CURSOS.get(ck,{}).get('nome',ck)}\n"]
        for cod, info in validas.items():
            materia   = info.get("materia", cod)
            professor = info.get("professor","A informar")
            horario   = info.get("horario","A informar")
            lines.append(f"• [{cod}] {materia}")
            lines.append(f"       {professor}")
            if horario != "-- A PREENCHER --":
                lines.append(f"        {horario}")
            lines.append("")
        lines.append("Para detalhes de uma disciplina específica, informe o código ou nome.")
        session["state"] = "idle"
        session["expects_more"] = True
        return ok(fmt(lines), sid)

    # =========================================================================
    # DETECÇÃO DE INTENÇÃO (estado idle)
    # =========================================================================

    # Matrícula — inscrição no Vestibular (separada da Rematrícula)
    if "matricula" in p and "rematricula" not in p:
        if not MATRICULA_ABERTA:
            return ok(
                "Matrícula na FATEC — Vestibular\n\n"
                "As inscrições para o Vestibular FATEC 2026-2 foram encerradas.\n\n"
                "Para ingressar na Fatec Zona Sul, você precisará aguardar as "
                "inscrições para o Vestibular 2027-1, com previsão de abertura "
                "em setembro de 2026 — confira o calendário acadêmico para as datas exatas.\n\n"
                "Fique atento ao site oficial da FATEC e ao portal acadêmico "
                "para não perder o prazo de inscrição!\n\n"
                "Já é aluno e quer fazer a rematrícula? É só digitar 'rematrícula'.", sid)
        # ── TESTER MODE: inscrições abertas ──────────────────────────────────
        session["state"] = "matricula.course"
        return ok(
            "Inscrição para o Vestibular FATEC — Modo Tester\n\n"
            "As inscrições estão abertas! Vamos iniciar a sua inscrição.\n\n"
            "Qual curso você deseja cursar?" + MENU_CURSOS, sid)

    # Rematrícula — alunos veteranos com RA
    if "rematricula" in p:
        session["state"] = "rematricula.ra"
        return ok(
            "Rematrícula — Alunos Veteranos\n\n"
            "Vou te ajudar com a rematrícula. Vamos começar pela identificação.\n\n"
            "Por favor, informe seu RA (Registro Acadêmico):", sid)

    # Trancamento
    if any(k in p for k in ["trancamento","trancar"]):
        info = dados.get("trancamento",{})
        lines = ["Trancamento de vínculo acadêmico:\n", info.get("descricao","")]
        if regras := info.get("regras"):
            lines.append("\nRegras importantes:")
            for r in regras: lines.append(f"• {r}")
        lines.append("\nDeseja iniciar o pedido de trancamento? (sim/não)")
        session["state"] = "trancamento.await_confirm"
        return ok(fmt(lines), sid)

    # Documentos
    if any(k in p for k in ["documento","historico","declaracao","comprovante"]):
        info  = dados.get("documentos",{})
        nomes = {"historico_escolar":"Histórico Escolar",
                 "declaracao_matricula":"Declaração de Matrícula",
                 "ementa_disciplina":"Ementa de Disciplina"}
        lines = ["Documentos disponíveis:\n"]
        for key,val in info.items():
            nome  = nomes.get(key, key.replace("_"," ").capitalize())
            canal = val.get("canal","") if isinstance(val,dict) else ""
            prazo = val.get("prazo","") if isinstance(val,dict) else ""
            lines += [f" {nome}", f"   Canal: {canal}", f"   Prazo: {prazo}\n"]
        lines.append("Deseja instruções para solicitar? (sim/não)")
        session["state"] = "documentos.await_confirm"
        return ok(fmt(lines), sid)

    # Transferência de horário
    if any(k in p for k in ["transferencia","transferir","mudar horario","mudar turno","trocar turno","trocar horario"]):
        info  = dados.get("transferencia_horario",{})
        lines = ["Transferência de turno/horário de aula:\n", info.get("descricao","")]
        if regras := info.get("regras"):
            lines.append("\nRegras:")
            for r in regras: lines.append(f"• {r}")
        lines.append("\nDeseja iniciar sua solicitação? (sim/não)")
        session["state"] = "trhorario.await_confirm"
        return ok(fmt(lines), sid)

    # Calendário acadêmico
    if any(k in p for k in ["calendario","calend","datas","data importante"]):
        session["state"] = "calendario.await_mes"
        return ok(
            "Calendário Acadêmico da Fatec Zona Sul\n\n"
            "Tenho as datas de Janeiro a Dezembro de 2026.\n\n"
            "Qual mês você deseja consultar?", sid)

    # Estágio
    if any(k in p for k in ["estagio","estagiar","estagios","estágio"]):
        info = dados.get("estagio",{})
        lines = [f" {info.get('descricao','Estágios')}\n",
                 "Você quer informações sobre:\n",
                 "1⃣ Estágio Obrigatório (curricular)",
                 "2⃣ Estágio Não Obrigatório (extracurricular)\n",
                 "Digite 1, 2 ou o nome do tipo."]
        session["state"] = "estagio.await_tipo"
        return ok(fmt(lines), sid)

    # Disciplinas — busca direta primeiro
    all_disc = dados.get("informacoes",{}).get("disciplinas",{})
    resultado = buscar_disc_global(p, all_disc)
    if resultado:
        cod, info, curso, sem = resultado
        materia   = info.get("materia", cod)
        professor = info.get("professor","Não informado")
        horario   = info.get("horario","-- A PREENCHER --")
        sala      = info.get("sala","-- A PREENCHER --")
        if any(k in p for k in ["professor","quem","ministra","leciona","responsavel"]):
            return ok(f" {materia} ({cod})\n{professor}", sid)
        if any(k in p for k in ["horario","hora","aula"]):
            return ok(f" {materia} ({cod}): {horario}", sid)
        if any(k in p for k in ["ementa","conteudo","objetivo"]):
            return ok("Solicite a ementa na secretaria presencial ou por e-mail institucional. Prazo: até 5 dias úteis.", sid)
        return ok(
            f" {materia}\nCódigo: {cod} | Curso: {curso} | {sem}º Semestre\n"
            f"{professor}\n Horário: {horario} |  Sala: {sala}\n\n"
            "Para a ementa, acesse a secretaria ou envie e-mail institucional.", sid)

    # Disciplinas — fluxo por curso/semestre
    if any(k in p for k in ["disciplina","grade","materia","materias","professor","professores","disciplinas"]):
        lines = ["Sobre qual curso você deseja consultar as disciplinas?\n"]
        for ck,cv in CURSOS.items(): lines.append(f"• {cv['nome']}")
        lines.append("\nDigite o nome ou a sigla.")
        session["state"] = "disciplinas.await_curso"
        return ok(fmt(lines), sid)

    # ── Motor LLM Generativo (open source) ──────────────────────────────────────
    # Aqui chegam todas as perguntas que não foram tratadas pelos fluxos estruturados.
    # O LLM recebe: identidade da FATMA + dados.json completo + histórico da sessão.
    resposta_llm = await consultar_llm(dados, request.pergunta, history)
    if resposta_llm:
        # Salva no histórico para manter contexto nas próximas trocas
        history.append({"role": "user",      "content": request.pergunta})
        history.append({"role": "assistant", "content": resposta_llm})
        # Mantém apenas as últimas N mensagens para não estourar o contexto
        session["history"] = history[-LLM_HISTORY_MAX:]
        return ok(resposta_llm, sid, f"llm_{LLM_PROVIDER}")

    # ── Fallback estático (LLM indisponível) ─────────────────────────────────────
    return ok(MENU_AJUDA.replace("Claro! ", "Não identifiquei sua solicitação. "), sid, "fallback_local")


@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "version":      "5.0.0",
        "llm_provider": LLM_PROVIDER,
        "llm_model":    OLLAMA_MODEL if LLM_PROVIDER == "ollama" else
                        GROQ_MODEL  if LLM_PROVIDER == "groq"   else HF_MODEL,
    }
