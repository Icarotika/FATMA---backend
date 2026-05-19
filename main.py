"""Backend reestruturado para conversação contínua orientada por `dados.json`.

Principais características:
- máquina de estados por sessão para fluxos (ex.: matrícula)
- respostas geradas localmente com templates naturais (sem devolver JSON cru)
- lembrete de contexto por sessão para evitar repetições e fluxos mortos
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

BASE_DIR = Path(__file__).resolve().parent.parent
DADOS_PATH = BASE_DIR / "backend/dados.json"

app = FastAPI(title="FATMA — Assistente Acadêmica", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    pergunta: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    resposta: str
    modo: str
    session_id: str | None = None


def carregar_dados_secretaria() -> dict[str, Any]:
    if not DADOS_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DADOS_PATH}")

    with DADOS_PATH.open("r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def pretty_list(items: list[str]) -> str:
    if not items:
        return "Nenhum item encontrado."
    if len(items) == 1:
        return items[0]
    return "; ".join(items[:-1]) + " e " + items[-1]


def gen_templates() -> dict[str, list[str]]:
    return {
        "procedimento_intro": [
            "Segue abaixo o procedimento resumido:",
            "Veja como funciona, passo a passo:",
            "Aqui está o procedimento de forma rápida e clara:"
        ],
        "ask_confirm": [
            "Deseja que eu te ajude a iniciar esse processo agora? (sim/não)",
            "Quer que eu te auxilie com esse procedimento agora?"
        ],
        "ask_course": [
            "Qual curso você quer cursar?",
            "Em qual curso você deseja se matricular?"
        ],
        "ask_shift": [
            "Qual turno prefere (matutino, vespertino, noturno)?",
            "Em qual turno você prefere estudar?"
        ],
        "confirming": [
            "Perfeito — vou confirmar os dados com você. Confirme: curso {curso}, turno {turno}. Está correto? (sim/não)",
            "Só para confirmar: curso {curso} no turno {turno}. Posso prosseguir? (sim/não)"
        ],
        "closing_next": [
            "Ótimo — prossigo com as próximas instruções.",
            "Certo, agora sigo para os próximos passos e informo você."]
    }


TEMPLATES = gen_templates()


SESSIONS: dict[str, dict[str, Any]] = {}


def is_affirmative(text: str) -> bool:
    text = text.lower()
    return bool(re.search(r"\b(sim|claro|quero|posso|fa[çc]a|okay|ok|confirmo|vamos|pode)\b", text))


def is_negative(text: str) -> bool:
    text = text.lower()
    return bool(re.search(r"\b(n[oã]o|nao|depois|mais tarde|agora não|agora nao|não agora)\b", text))


async def consultar_openai(prompt_base: str, pergunta: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": prompt_base},
            {"role": "user", "content": pergunta},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def format_response_blocks(title: str | None, lines: list[str]) -> str:
    parts: list[str] = []
    if title:
        parts.append(f"{title}")
    parts.extend(lines)
    return "\n".join(parts)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        dados = carregar_dados_secretaria()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="dados.json inválido") from exc

    session_id = request.session_id or uuid.uuid4().hex
    session = SESSIONS.setdefault(session_id, {"state": "idle", "context": {}})

    pergunta = request.pergunta.strip()
    p_norm = strip_accents(pergunta.lower())

    # If session expects a specific step, continue the flow
    state = session.get("state", "idle")

    # ---- Matrícula flow ----
    if state.startswith("matricula"):
        # matricula steps: procedimento -> confirm -> documentos -> curso -> turno -> confirm_data -> done
        step = state.split(".")[-1]

        if step == "await_confirm":
            if is_affirmative(p_norm):
                # provide documents and go to course selection
                requisitos = dados.get("matrícula", {}).get("requisitos", [])
                lines = ["Para prosseguir, reúna os seguintes documentos:", ""]
                for r in requisitos:
                    lines.append(f"- {r}")
                lines.append("")
                lines.append(random.choice(TEMPLATES["ask_course"]))
                session["state"] = "matricula.course"
                return ChatResponse(resposta=format_response_blocks("", lines), modo="conversacional", session_id=session_id)

            if is_negative(p_norm):
                session["state"] = "idle"
                return ChatResponse(resposta="Tudo bem — quando quiser posso retomar esse assunto. Posso ajudar em outra coisa?", modo="conversacional", session_id=session_id)

            return ChatResponse(resposta="Desculpe, não entendi. Você deseja iniciar agora o processo de matrícula? (sim/não)", modo="conversacional", session_id=session_id)

        if step == "course":
            # save course and ask shift
            session["context"]["curso"] = pergunta
            session["state"] = "matricula.turno"
            return ChatResponse(resposta=random.choice(TEMPLATES["ask_shift"]), modo="conversacional", session_id=session_id)

        if step == "turno":
            session["context"]["turno"] = pergunta
            session["state"] = "matricula.confirm_data"
            curso = session["context"].get("curso", "(não informado)")
            turno = session["context"].get("turno", "(não informado)")
            tpl = random.choice(TEMPLATES["confirming"]) .format(curso=curso, turno=turno)
            return ChatResponse(resposta=tpl, modo="conversacional", session_id=session_id)

        if step == "confirm_data":
            if is_affirmative(p_norm):
                session["state"] = "idle"
                return ChatResponse(resposta="Perfeito — matrícula solicitada. Em breve você receberá informações por e-mail com os próximos passos.", modo="conversacional", session_id=session_id)
            if is_negative(p_norm):
                session["state"] = "matricula.course"
                return ChatResponse(resposta="Ok — vamos recomeçar. Qual curso você deseja?", modo="conversacional", session_id=session_id)
            return ChatResponse(resposta="Não entendi — confirme por favor: os dados estão corretos? (sim/não)", modo="conversacional", session_id=session_id)

    # ---- Detect intent ----
    # matrícula intent
    if any(k in p_norm for k in ["matricula", "matrícula", "rematricula", "rematrícula"]):
        procedimento = dados.get("matrícula", {}).get("procedimento") or [dados.get("matrícula", {}).get("descricao", "")]
        lines = [random.choice(TEMPLATES["procedimento_intro"]), ""]
        if isinstance(procedimento, list):
            for i, passo in enumerate(procedimento, 1):
                lines.append(f"{i}. {passo}")
        else:
            lines.append(str(procedimento))

        lines.append("")
        lines.append(random.choice(TEMPLATES["ask_confirm"]))
        session["state"] = "matricula.await_confirm"
        return ChatResponse(resposta=format_response_blocks("", lines), modo="conversacional", session_id=session_id)

    # trancamento intent
    if any(k in p_norm for k in ["trancamento", "trancar"]):
        info = dados.get("trancamento", {})
        lines = ["Trancamento de vínculo — resumo:" , ""]
        if desc := info.get("descricao"):
            lines.append(desc)
        if regras := info.get("regras"):
            lines.append("")
            lines.append("Regras importantes:")
            for r in regras:
                lines.append(f"- {r}")
        lines.append("")
        lines.append("Deseja que eu inicie o pedido de trancamento para você? (sim/não)")
        session["state"] = "trancamento.await_confirm"
        return ChatResponse(resposta=format_response_blocks("", lines), modo="conversacional", session_id=session_id)

    # documentos intent
    if any(k in p_norm for k in ["documento", "documentos", "histórico", "historico", "comprovante"]):
        info = dados.get("documentos", {})
        lines = ["Documentos e canais:", ""]
        for key, val in info.items():
            canal = val.get("canal") if isinstance(val, dict) else ""
            prazo = val.get("prazo") if isinstance(val, dict) else ""
            lines.append(f"• {key.replace('_', ' ').capitalize()}: {canal} — {prazo}")
        lines.append("")
        lines.append("Deseja que eu gere instruções para solicitar algum documento específico? (sim/não)")
        session["state"] = "documentos.await_confirm"
        return ChatResponse(resposta=format_response_blocks("", lines), modo="conversacional", session_id=session_id)

    # prazos intent
    if "prazo" in p_norm or "prazos" in p_norm:
        info = dados.get("prazos", {})
        lines = ["Principais prazos:", ""]
        for k, v in info.items():
            lines.append(f"• {k.replace('_', ' ').capitalize()}: {v}")
        return ChatResponse(resposta=format_response_blocks("", lines), modo="conversacional", session_id=session_id)

    # disciplinas (professor, conteúdo, horário, pré-requisitos, duração)
    disciplinas = dados.get("disciplinas", {})
    for nome, info in disciplinas.items():
        chave = strip_accents(nome.lower())
        if chave in p_norm:
            if "professor" in p_norm or "quem" in p_norm:
                resp = f"Professor da disciplina {info.get('nome', nome)}: {info.get('professor', 'Não informado')}"
                return ChatResponse(resposta=resp, modo="conversacional", session_id=session_id)
            if any(k in p_norm for k in ["conteudo", "ementa", "objetivo"]):
                resp = f"Ementa / Objetivos de {info.get('nome', nome)}:\n{info.get('conteudo', 'Não informado')}"
                return ChatResponse(resposta=resp, modo="conversacional", session_id=session_id)
            if any(k in p_norm for k in ["horario", "hora"]):
                resp = f"Horário de {info.get('nome', nome)}: {info.get('horario', 'Não informado')}"
                return ChatResponse(resposta=resp, modo="conversacional", session_id=session_id)
            if any(k in p_norm for k in ["pre-requisito", "prerequisito", "pré-requisito"]):
                pre = info.get("pre_requisitos") or []
                resp = "Pré-requisitos: " + (pretty_list(pre) if isinstance(pre, list) else str(pre))
                return ChatResponse(resposta=resp, modo="conversacional", session_id=session_id)
            if "duracao" in p_norm or "duração" in p_norm:
                resp = f"Duração: {info.get('duracao', 'Não informado')}"
                return ChatResponse(resposta=resp, modo="conversacional", session_id=session_id)

    # tentativa com OpenAI (se disponível) para respostas mais naturais
    prompt_base = "Contexto: " + json.dumps(dados, ensure_ascii=False)
    resposta_openai = await consultar_openai(prompt_base, request.pergunta)
    if resposta_openai:
        # store last assistant message to avoid exact repeats
        session.setdefault("history", []).append({"user": pergunta, "bot": resposta_openai})
        return ChatResponse(resposta=resposta_openai, modo="openai", session_id=session_id)

    # fallback local generoso (não devolver JSON cru)
    generic = (
        "Desculpe — não identifiquei a intenção específica. Posso ajudar com matrícula, trancamento, documentos, prazos ou informações sobre disciplinas."
    )
    return ChatResponse(resposta=generic, modo="fallback_local", session_id=session_id)
