"""Backend FATMA — Assistente Acadêmica v3.0

Correções aplicadas:
- caminho correto para disciplinas: dados["informacoes"]["disciplinas"]
- estados trancamento.await_confirm e documentos.await_confirm agora são tratados
- busca de disciplina por código E por nome da matéria
- fallback via API Anthropic (Claude) no lugar da chave OpenAI inválida
- sistema de contexto de sessão mais robusto
"""

from __future__ import annotations

import json
import os
import random
import re
import uuid
import unicodedata
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DADOS_PATH = BASE_DIR / "dados.json"

app = FastAPI(title="FATMA — Assistente Acadêmica", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    pergunta: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    resposta: str
    modo: str
    session_id: str | None = None


# ── Data helpers ─────────────────────────────────────────────────────────────

def carregar_dados() -> dict[str, Any]:
    if not DADOS_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DADOS_PATH}")
    with DADOS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalizar(text: str) -> str:
    return strip_accents(text.lower().strip())


def pretty_list(items: list[str]) -> str:
    if not items:
        return "Nenhum item encontrado."
    if len(items) == 1:
        return items[0]
    return "; ".join(items[:-1]) + " e " + items[-1]


def format_blocks(title: str | None, lines: list[str]) -> str:
    parts: list[str] = []
    if title:
        parts.append(title)
    parts.extend(lines)
    return "\n".join(parts)


# ── Templates ────────────────────────────────────────────────────────────────

TEMPLATES: dict[str, list[str]] = {
    "procedimento_intro": [
        "Segue o procedimento resumido:",
        "Veja como funciona, passo a passo:",
        "Aqui está o procedimento de forma rápida e clara:",
    ],
    "ask_confirm": [
        "Deseja que eu te ajude a iniciar esse processo agora? (sim/não)",
        "Quer que eu te auxilie com esse procedimento? (sim/não)",
    ],
    "ask_course": [
        "Qual curso você quer cursar?",
        "Em qual curso você deseja se matricular?",
    ],
    "ask_shift": [
        "Qual turno prefere — matutino, vespertino ou noturno?",
        "Em qual turno você prefere estudar?",
    ],
    "confirming": [
        "Perfeito! Confirmando: curso {curso}, turno {turno}. Está correto? (sim/não)",
        "Só para confirmar: curso {curso} no turno {turno}. Posso prosseguir? (sim/não)",
    ],
}

# ── Session store ─────────────────────────────────────────────────────────────

SESSIONS: dict[str, dict[str, Any]] = {}


def is_affirmative(text: str) -> bool:
    return bool(re.search(
        r"\b(sim|claro|quero|posso|fa[çc]a|okay|ok|confirmo|vamos|pode|isso)\b",
        text.lower()
    ))


def is_negative(text: str) -> bool:
    return bool(re.search(
        r"\b(n[oã]o|nao|depois|mais tarde|agora não|agora nao|não agora)\b",
        text.lower()
    ))


# ── Anthropic fallback ───────────────────────────────────────────────────────

async def consultar_claude(contexto: str, pergunta: str) -> str | None:
    """Chama a API da Anthropic (Claude) como fallback inteligente."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    system_prompt = (
        "Você é FATMA, assistente acadêmica virtual da Fatec Zona Sul. "
        "Responda sempre em português, de forma clara e amigável. "
        "Use apenas as informações do contexto abaixo. "
        "Se não souber a resposta, diga que pode ajudar com matrícula, "
        "trancamento, documentos, prazos ou disciplinas.\n\n"
        f"CONTEXTO:\n{contexto}"
    )
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 512,
        "system": system_prompt,
        "messages": [{"role": "user", "content": pergunta}],
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"].strip()
    except Exception:
        return None


# ── Disciplina helpers ───────────────────────────────────────────────────────

def buscar_disciplina(p_norm: str, disciplinas: dict) -> tuple[str, dict] | None:
    """
    Busca disciplina por código (ex: IAL027) OU por parte do nome da matéria.
    Retorna (codigo, info) ou None.
    """
    for codigo, info in disciplinas.items():
        # busca por código direto (ex: "ial027")
        if normalizar(codigo) in p_norm:
            return codigo, info
        # busca por palavras do nome da matéria (ex: "algoritmos")
        materia_norm = normalizar(info.get("materia", ""))
        palavras = [w for w in materia_norm.split() if len(w) > 3]
        if any(palavra in p_norm for palavra in palavras):
            return codigo, info
    return None


# ── Main endpoint ────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        dados = carregar_dados()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="dados.json inválido") from exc

    session_id = request.session_id or uuid.uuid4().hex
    session = SESSIONS.setdefault(session_id, {"state": "idle", "context": {}})

    pergunta = request.pergunta.strip()
    p_norm = normalizar(pergunta)
    state = session.get("state", "idle")

    # ── Fluxo: Matrícula ─────────────────────────────────────────────────────

    if state.startswith("matricula."):
        step = state.split(".", 1)[1]

        if step == "await_confirm":
            if is_affirmative(p_norm):
                requisitos = dados.get("matrícula", {}).get("requisitos", [])
                lines = ["Para prosseguir, reúna os seguintes documentos:\n"]
                for r in requisitos:
                    lines.append(f"• {r}")
                lines.append("")
                lines.append(random.choice(TEMPLATES["ask_course"]))
                session["state"] = "matricula.course"
                return ChatResponse(
                    resposta=format_blocks(None, lines),
                    modo="conversacional",
                    session_id=session_id,
                )
            if is_negative(p_norm):
                session["state"] = "idle"
                return ChatResponse(
                    resposta="Tudo bem! Quando quiser retomamos. Posso ajudar em outra coisa?",
                    modo="conversacional",
                    session_id=session_id,
                )
            return ChatResponse(
                resposta="Desculpe, não entendi. Deseja iniciar o processo de matrícula agora? (sim/não)",
                modo="conversacional",
                session_id=session_id,
            )

        if step == "course":
            session["context"]["curso"] = pergunta
            session["state"] = "matricula.turno"
            return ChatResponse(
                resposta=random.choice(TEMPLATES["ask_shift"]),
                modo="conversacional",
                session_id=session_id,
            )

        if step == "turno":
            session["context"]["turno"] = pergunta
            session["state"] = "matricula.confirm_data"
            curso = session["context"].get("curso", "(não informado)")
            turno = session["context"].get("turno", "(não informado)")
            tpl = random.choice(TEMPLATES["confirming"]).format(curso=curso, turno=turno)
            return ChatResponse(resposta=tpl, modo="conversacional", session_id=session_id)

        if step == "confirm_data":
            if is_affirmative(p_norm):
                session["state"] = "idle"
                session["context"] = {}
                return ChatResponse(
                    resposta="✅ Matrícula solicitada com sucesso! Em breve você receberá um e-mail com os próximos passos. Há mais alguma coisa em que posso ajudar?",
                    modo="conversacional",
                    session_id=session_id,
                )
            if is_negative(p_norm):
                session["state"] = "matricula.course"
                return ChatResponse(
                    resposta="Ok, vamos recomeçar. Qual curso você deseja?",
                    modo="conversacional",
                    session_id=session_id,
                )
            return ChatResponse(
                resposta="Não entendi — os dados estão corretos? (sim/não)",
                modo="conversacional",
                session_id=session_id,
            )

    # ── Fluxo: Trancamento ───────────────────────────────────────────────────

    if state == "trancamento.await_confirm":
        if is_affirmative(p_norm):
            prazo = dados.get("trancamento", {}).get("prazo", "consulte o portal")
            session["state"] = "idle"
            return ChatResponse(
                resposta=(
                    "Para solicitar o trancamento:\n\n"
                    "1. Acesse o portal acadêmico\n"
                    "2. Vá em Secretaria → Trancamento de Matrícula\n"
                    "3. Confirme a solicitação\n\n"
                    f"⚠️ Prazo: {prazo}.\n\n"
                    "Deseja ajuda com mais alguma coisa?"
                ),
                modo="conversacional",
                session_id=session_id,
            )
        if is_negative(p_norm):
            session["state"] = "idle"
            return ChatResponse(
                resposta="Tudo bem! Se precisar, é só chamar. Posso ajudar em outra coisa?",
                modo="conversacional",
                session_id=session_id,
            )
        return ChatResponse(
            resposta="Deseja que eu inicie o pedido de trancamento? (sim/não)",
            modo="conversacional",
            session_id=session_id,
        )

    # ── Fluxo: Documentos ────────────────────────────────────────────────────

    if state == "documentos.await_confirm":
        if is_affirmative(p_norm):
            session["state"] = "idle"
            return ChatResponse(
                resposta=(
                    "Para solicitar documentos:\n\n"
                    "📄 Histórico escolar e Declaração de matrícula:\n"
                    "→ Acesse o portal acadêmico e emita em PDF imediatamente.\n\n"
                    "📋 Ementa de disciplina:\n"
                    "→ Envie e-mail para a secretaria ou compareça presencialmente.\n"
                    "→ Prazo: até 5 dias úteis.\n\n"
                    "Precisa de mais alguma coisa?"
                ),
                modo="conversacional",
                session_id=session_id,
            )
        if is_negative(p_norm):
            session["state"] = "idle"
            return ChatResponse(
                resposta="Tudo bem! Fico à disposição. Posso ajudar em outra coisa?",
                modo="conversacional",
                session_id=session_id,
            )
        return ChatResponse(
            resposta="Deseja instruções para solicitar um documento específico? (sim/não)",
            modo="conversacional",
            session_id=session_id,
        )

    # ── Detecção de intenção ─────────────────────────────────────────────────

    # Matrícula
    if any(k in p_norm for k in ["matricula", "rematricula"]):
        procedimento = dados.get("matrícula", {}).get("procedimento", [])
        descricao = dados.get("matrícula", {}).get("descricao", "")
        lines = [random.choice(TEMPLATES["procedimento_intro"]), ""]
        if procedimento:
            for i, passo in enumerate(procedimento, 1):
                lines.append(f"{i}. {passo}")
        else:
            lines.append(descricao)
        lines.append("")
        lines.append(random.choice(TEMPLATES["ask_confirm"]))
        session["state"] = "matricula.await_confirm"
        return ChatResponse(resposta=format_blocks(None, lines), modo="conversacional", session_id=session_id)

    # Trancamento
    if any(k in p_norm for k in ["trancamento", "trancar"]):
        info = dados.get("trancamento", {})
        lines = ["Trancamento de vínculo acadêmico:\n"]
        if desc := info.get("descricao"):
            lines.append(desc)
        if regras := info.get("regras"):
            lines.append("\nRegras importantes:")
            for r in regras:
                lines.append(f"• {r}")
        lines.append("")
        lines.append("Deseja que eu inicie o pedido de trancamento para você? (sim/não)")
        session["state"] = "trancamento.await_confirm"
        return ChatResponse(resposta=format_blocks(None, lines), modo="conversacional", session_id=session_id)

    # Documentos
    if any(k in p_norm for k in ["documento", "historico", "declaracao", "comprovante", "ementa"]):
        info = dados.get("documentos", {})
        lines = ["Documentos disponíveis e como obtê-los:\n"]
        nomes = {
            "historico_escolar": "Histórico Escolar",
            "declaracao_matricula": "Declaração de Matrícula",
            "ementa_disciplina": "Ementa de Disciplina",
        }
        for key, val in info.items():
            nome = nomes.get(key, key.replace("_", " ").capitalize())
            canal = val.get("canal", "") if isinstance(val, dict) else ""
            prazo = val.get("prazo", "") if isinstance(val, dict) else ""
            lines.append(f"📄 {nome}")
            lines.append(f"   Canal: {canal}")
            lines.append(f"   Prazo: {prazo}\n")
        lines.append("Deseja instruções para solicitar algum documento? (sim/não)")
        session["state"] = "documentos.await_confirm"
        return ChatResponse(resposta=format_blocks(None, lines), modo="conversacional", session_id=session_id)

    # Prazos
    if any(k in p_norm for k in ["prazo", "prazos", "quando", "data"]):
        info = dados.get("prazos", {})
        nomes = {
            "rematricula": "Rematrícula",
            "ajuste_disciplinas": "Ajuste de Disciplinas",
            "trancamento_semestre": "Trancamento de Semestre",
            "solicitacao_documentos_especiais": "Solicitação de Documentos Especiais",
        }
        lines = ["📅 Principais prazos do semestre:\n"]
        for k, v in info.items():
            nome = nomes.get(k, k.replace("_", " ").capitalize())
            lines.append(f"• {nome}: {v}")
        return ChatResponse(resposta=format_blocks(None, lines), modo="conversacional", session_id=session_id)

    # Disciplinas — FIX: caminho correto é dados["informacoes"]["disciplinas"]
    disciplinas = dados.get("informacoes", {}).get("disciplinas", {})
    resultado = buscar_disciplina(p_norm, disciplinas)
    if resultado:
        codigo, info = resultado
        materia = info.get("materia", codigo)
        professor = info.get("professor", "Não informado")

        if any(k in p_norm for k in ["professor", "quem", "ministra", "leciona"]):
            return ChatResponse(
                resposta=f"👨‍🏫 {materia} ({codigo})\nProfessor(a): {professor}",
                modo="conversacional",
                session_id=session_id,
            )
        if any(k in p_norm for k in ["ementa", "conteudo", "objetivo"]):
            return ChatResponse(
                resposta=f"A ementa de {materia} deve ser solicitada na secretaria presencial ou por e-mail institucional. Prazo: até 5 dias úteis.",
                modo="conversacional",
                session_id=session_id,
            )
        # resposta geral sobre a disciplina
        return ChatResponse(
            resposta=(
                f"📚 Disciplina: {materia}\n"
                f"Código: {codigo}\n"
                f"Professor(a): {professor}\n\n"
                "Para obter a ementa, acesse a secretaria presencial ou envie e-mail institucional."
            ),
            modo="conversacional",
            session_id=session_id,
        )

    # ── Fallback: Claude (Anthropic) ─────────────────────────────────────────
    contexto = json.dumps(dados, ensure_ascii=False, indent=2)
    resposta_ia = await consultar_claude(contexto, request.pergunta)
    if resposta_ia:
        return ChatResponse(resposta=resposta_ia, modo="claude_ai", session_id=session_id)

    # ── Fallback local ────────────────────────────────────────────────────────
    return ChatResponse(
        resposta=(
            "Não identifiquei sua solicitação. Posso ajudar com:\n\n"
            "• Matrícula e rematrícula\n"
            "• Trancamento de curso\n"
            "• Documentos (histórico, declarações)\n"
            "• Prazos do semestre\n"
            "• Informações sobre disciplinas\n\n"
            "O que você precisa?"
        ),
        modo="fallback_local",
        session_id=session_id,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "3.0.0"}
