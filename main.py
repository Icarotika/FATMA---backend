"""Backend FATMA — Assistente Acadêmica v4.0 (Alpha 0.3.1)

Patch Notes:
- Removida conversação "Prazos"
- Adicionadas: Transferência de Horário, Calendário Acadêmico, Estágio, Disciplinas melhorada
- Matrícula: validação de curso e turno com regras por curso; coleta de RA
- Trancamento: coleta de RA do aluno
- Transferência de horário: coleta de RA, turno atual e turno desejado
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

BASE_DIR  = Path(__file__).resolve().parent
DADOS_PATH = BASE_DIR / "dados.json"

app = FastAPI(title="FATMA — Assistente Acadêmica", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    pergunta: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None

class ChatResponse(BaseModel):
    resposta: str
    modo: str
    session_id: str | None = None

# ── Data helpers ──────────────────────────────────────────────────────────────

def carregar_dados() -> dict[str, Any]:
    if not DADOS_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DADOS_PATH}")
    with DADOS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def normalizar(text: str) -> str:
    return strip_accents(text.lower().strip())

def format_blocks(title: str | None, lines: list[str]) -> str:
    parts: list[str] = ([title] if title else []) + lines
    return "\n".join(parts)

def ok(msg: str, session_id: str, mode: str = "conversacional") -> ChatResponse:
    return ChatResponse(resposta=msg, modo=mode, session_id=session_id)

# ── Cursos e turnos válidos ───────────────────────────────────────────────────

CURSOS: dict[str, dict] = {
    "ads": {
        "nome": "Análise e Desenvolvimento de Sistemas (ADS)",
        "turnos": ["matutino", "noturno"],
        "turnos_msg": "Matutino ou Noturno",
    },
    "dsm": {
        "nome": "Desenvolvimento de Software Multiplataforma (DSM)",
        "turnos": ["vespertino"],
        "turnos_msg": "Vespertino (único turno disponível)",
    },
    "logistica": {
        "nome": "Logística",
        "turnos": ["vespertino", "noturno"],
        "turnos_msg": "Vespertino ou Noturno",
    },
    "gestao": {
        "nome": "Gestão Empresarial",
        "turnos": ["matutino", "vespertino", "ead"],
        "turnos_msg": "Matutino, Vespertino ou EaD",
    },
}

# Aliases para reconhecer o que o usuário digitou
CURSO_ALIASES: dict[str, str] = {
    "ads": "ads", "analise": "ads", "analise e desenvolvimento": "ads",
    "desenvolvimento de sistemas": "ads",
    "dsm": "dsm", "multiplataforma": "dsm", "software multiplataforma": "dsm",
    "desenvolvimento de software": "dsm",
    "logistica": "logistica", "logísticas": "logistica", "logstica": "logistica",
    "gestao": "gestao", "gestão": "gestao", "empresarial": "gestao",
    "gestao empresarial": "gestao",
}

TURNO_ALIASES: dict[str, str] = {
    "matutino": "matutino", "manha": "matutino", "manha": "matutino",
    "vespertino": "vespertino", "tarde": "vespertino",
    "noturno": "noturno", "noite": "noturno",
    "ead": "ead", "distancia": "ead", "online": "ead", "a distancia": "ead",
    "ensino a distancia": "ead",
}

MENU_CURSOS = (
    "\nCursos disponíveis:\n"
    "  • Análise e Desenvolvimento de Sistemas (ADS)\n"
    "  • Desenvolvimento de Software Multiplataforma (DSM)\n"
    "  • Logística\n"
    "  • Gestão Empresarial\n"
    "\nDigite o nome ou a sigla do curso."
)

def detectar_curso(p_norm: str) -> str | None:
    # tenta alias mais longo primeiro
    for alias in sorted(CURSO_ALIASES, key=len, reverse=True):
        if alias in p_norm:
            return CURSO_ALIASES[alias]
    return None

def detectar_turno(p_norm: str) -> str | None:
    for alias, key in TURNO_ALIASES.items():
        if alias in p_norm:
            return key
    return None

# ── Session store ─────────────────────────────────────────────────────────────

SESSIONS: dict[str, dict[str, Any]] = {}

def is_affirmative(text: str) -> bool:
    return bool(re.search(r"\b(sim|claro|quero|posso|fa[çc]a|okay|ok|confirmo|vamos|pode|isso|certo)\b", text.lower()))

def is_negative(text: str) -> bool:
    return bool(re.search(r"\b(n[aã]o|nao|depois|mais tarde|agora não|agora nao|cancelar)\b", text.lower()))

# ── Anthropic fallback ────────────────────────────────────────────────────────

async def consultar_claude(contexto: str, pergunta: str) -> str | None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    system_prompt = (
        "Você é FATMA, assistente acadêmica virtual da Fatec Zona Sul. "
        "Responda sempre em português, de forma clara e amigável. "
        "Use apenas as informações do contexto abaixo. Se não souber, "
        "diga que pode ajudar com matrícula, trancamento, documentos, "
        "transferência de horário, estágio, calendário ou disciplinas.\n\n"
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
            return resp.json()["content"][0]["text"].strip()
    except Exception:
        return None

# ── Disciplina helpers ────────────────────────────────────────────────────────

def buscar_disciplina(p_norm: str, disciplinas: dict) -> tuple[str, dict] | None:
    for codigo, info in disciplinas.items():
        if normalizar(codigo) in p_norm:
            return codigo, info
        palavras = [w for w in normalizar(info.get("materia", "")).split() if len(w) > 3]
        if any(p in p_norm for p in palavras):
            return codigo, info
    return None

# ── Endpoint principal ────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        dados = carregar_dados()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="dados.json inválido") from exc

    sid     = request.session_id or uuid.uuid4().hex
    session = SESSIONS.setdefault(sid, {"state": "idle", "context": {}})
    pergunta = request.pergunta.strip()
    p_norm   = normalizar(pergunta)
    state    = session.get("state", "idle")
    ctx      = session["context"]

    # =========================================================================
    # FLUXO: MATRÍCULA
    # =========================================================================
    if state.startswith("matricula."):
        step = state.split(".", 1)[1]

        # ── Aguarda confirmação inicial ──
        if step == "await_confirm":
            if is_affirmative(p_norm):
                requisitos = dados.get("matrícula", {}).get("requisitos", [])
                lines = ["Ótimo! Antes de iniciarmos, reúna os seguintes documentos:\n"]
                for r in requisitos:
                    lines.append(f"• {r}")
                lines.append(MENU_CURSOS)
                session["state"] = "matricula.course"
                return ok(format_blocks(None, lines), sid)
            if is_negative(p_norm):
                session["state"] = "idle"
                return ok("Tudo bem! Quando quiser retomamos. Posso ajudar em outra coisa?", sid)
            return ok("Deseja iniciar o processo de matrícula agora? (sim/não)", sid)

        # ── Coleta e valida curso ──
        if step == "course":
            curso_key = detectar_curso(p_norm)
            if not curso_key:
                return ok(
                    "Não reconheci o curso digitado. Por favor, escolha uma das opções abaixo:"
                    + MENU_CURSOS, sid
                )
            ctx["curso_key"]  = curso_key
            ctx["curso_nome"] = CURSOS[curso_key]["nome"]
            turnos_msg        = CURSOS[curso_key]["turnos_msg"]

            # DSM tem turno único — pula pergunta e já confirma
            if len(CURSOS[curso_key]["turnos"]) == 1:
                turno_unico = CURSOS[curso_key]["turnos"][0].capitalize()
                ctx["turno"] = CURSOS[curso_key]["turnos"][0]
                session["state"] = "matricula.confirm_data"
                return ok(
                    f"Curso selecionado: {ctx['curso_nome']}\n"
                    f"O único turno disponível para esse curso é o {turno_unico}.\n\n"
                    f"Confirmar matrícula em {ctx['curso_nome']} — {turno_unico}? (sim/não)", sid
                )

            session["state"] = "matricula.turno"
            return ok(
                f"Curso selecionado: {ctx['curso_nome']} ✅\n\n"
                f"Turnos disponíveis: {turnos_msg}\n"
                "Qual turno você prefere?", sid
            )

        # ── Coleta e valida turno ──
        if step == "turno":
            turno_key   = detectar_turno(p_norm)
            curso_key   = ctx.get("curso_key", "")
            turnos_ok   = CURSOS.get(curso_key, {}).get("turnos", [])
            turnos_msg  = CURSOS.get(curso_key, {}).get("turnos_msg", "")

            if not turno_key or turno_key not in turnos_ok:
                turno_exibido = turno_key.capitalize() if turno_key else "informado"
                return ok(
                    f"O turno {turno_exibido} não está disponível para {ctx.get('curso_nome', 'este curso')}.\n"
                    f"Turnos válidos: {turnos_msg}\n\n"
                    "Qual você prefere?", sid
                )

            ctx["turno"] = turno_key
            session["state"] = "matricula.confirm_data"
            return ok(
                f"Perfeito! Confirmando:\n"
                f"📚 Curso: {ctx['curso_nome']}\n"
                f"🕐 Turno: {turno_key.capitalize()}\n\n"
                "Os dados estão corretos? (sim/não)", sid
            )

        # ── Confirmação dos dados ──
        if step == "confirm_data":
            if is_affirmative(p_norm):
                session["state"] = "matricula.ra"
                return ok(
                    "Dados confirmados! Agora preciso identificar seu cadastro.\n\n"
                    "Por favor, informe seu RA (Registro Acadêmico) ou CPF:", sid
                )
            if is_negative(p_norm):
                session["state"] = "matricula.course"
                ctx.clear()
                return ok("Sem problema! Vamos recomeçar.\n" + MENU_CURSOS, sid)
            return ok("Não entendi — os dados estão corretos? (sim/não)", sid)

        # ── Coleta de RA ──
        if step == "ra":
            ra = re.sub(r"\D", "", pergunta)  # extrai apenas dígitos
            if len(ra) < 5:
                return ok("Não consegui identificar o RA/CPF. Por favor, informe apenas os números.", sid)
            ctx["ra"] = ra
            curso_nome = ctx.get("curso_nome", "")
            turno      = ctx.get("turno", "").capitalize()
            session["state"] = "idle"
            session["context"] = {}
            return ok(
                f"✅ Solicitação de matrícula registrada!\n\n"
                f"📋 Resumo:\n"
                f"• RA/CPF: {ra}\n"
                f"• Curso: {curso_nome}\n"
                f"• Turno: {turno}\n\n"
                "Em breve você receberá um e-mail com as próximas instruções. "
                "Há mais alguma coisa em que posso ajudar?", sid
            )

    # =========================================================================
    # FLUXO: TRANCAMENTO
    # =========================================================================
    if state == "trancamento.await_confirm":
        if is_affirmative(p_norm):
            session["state"] = "trancamento.ra"
            return ok(
                "Para iniciar o processo, preciso identificar seu cadastro.\n\n"
                "Por favor, informe seu RA (Registro Acadêmico):", sid
            )
        if is_negative(p_norm):
            session["state"] = "idle"
            return ok("Tudo bem! Se precisar, é só chamar. Posso ajudar em outra coisa?", sid)
        return ok("Deseja que eu inicie o pedido de trancamento? (sim/não)", sid)

    if state == "trancamento.ra":
        ra = re.sub(r"\D", "", pergunta)
        if len(ra) < 5:
            return ok("Não consegui identificar o RA. Por favor, informe apenas os números.", sid)
        ctx["ra"] = ra
        prazo = dados.get("trancamento", {}).get("prazo", "consulte o portal")
        session["state"] = "idle"
        session["context"] = {}
        return ok(
            f"Solicitação registrada para o RA {ra}.\n\n"
            "Para concluir o trancamento:\n\n"
            "1. Acesse o portal acadêmico\n"
            "2. Vá em Secretaria → Trancamento de Matrícula\n"
            "3. Confirme a solicitação\n\n"
            f"⚠️ Prazo: {prazo}\n\n"
            "Deseja ajuda com mais alguma coisa?", sid
        )

    # =========================================================================
    # FLUXO: DOCUMENTOS
    # =========================================================================
    if state == "documentos.await_confirm":
        if is_affirmative(p_norm):
            session["state"] = "documentos.ra"
            return ok("Para registrar a solicitação, informe seu RA (Registro Acadêmico):", sid)
        if is_negative(p_norm):
            session["state"] = "idle"
            return ok("Tudo bem! Fico à disposição. Posso ajudar em outra coisa?", sid)
        return ok("Deseja instruções para solicitar um documento específico? (sim/não)", sid)

    if state == "documentos.ra":
        ra = re.sub(r"\D", "", pergunta)
        if len(ra) < 5:
            return ok("Não consegui identificar o RA. Por favor, informe apenas os números.", sid)
        session["state"] = "idle"
        session["context"] = {}
        return ok(
            f"Solicitação registrada para o RA {ra}.\n\n"
            "📄 Histórico Escolar e Declaração de Matrícula:\n"
            "→ Acesse o portal acadêmico e emita em PDF imediatamente.\n\n"
            "📋 Ementa de Disciplina:\n"
            "→ Envie e-mail para a secretaria ou compareça presencialmente.\n"
            "→ Prazo: até 5 dias úteis.\n\n"
            "Precisa de mais alguma coisa?", sid
        )

    # =========================================================================
    # FLUXO: TRANSFERÊNCIA DE HORÁRIO
    # =========================================================================
    if state == "trhorario.await_confirm":
        if is_affirmative(p_norm):
            session["state"] = "trhorario.ra"
            return ok(
                "Vou registrar sua solicitação de transferência de horário.\n\n"
                "Primeiro, informe seu RA (Registro Acadêmico):", sid
            )
        if is_negative(p_norm):
            session["state"] = "idle"
            return ok("Tudo bem! Quando precisar, é só chamar. Posso ajudar com outra coisa?", sid)
        return ok("Deseja iniciar a solicitação de transferência de horário? (sim/não)", sid)

    if state == "trhorario.ra":
        ra = re.sub(r"\D", "", pergunta)
        if len(ra) < 5:
            return ok("Não consegui identificar o RA. Por favor, informe apenas os números.", sid)
        ctx["ra"] = ra
        session["state"] = "trhorario.turno_atual"
        return ok(
            f"RA {ra} registrado. ✅\n\n"
            "Agora, qual é o seu turno atual?\n"
            "(Matutino / Vespertino / Noturno)", sid
        )

    if state == "trhorario.turno_atual":
        turno = detectar_turno(p_norm)
        if not turno or turno == "ead":
            return ok("Não reconheci o turno. Informe: Matutino, Vespertino ou Noturno.", sid)
        ctx["turno_atual"] = turno
        session["state"] = "trhorario.turno_desejado"
        return ok(
            f"Turno atual: {turno.capitalize()} ✅\n\n"
            "Para qual turno deseja transferir?\n"
            "(Matutino / Vespertino / Noturno)", sid
        )

    if state == "trhorario.turno_desejado":
        turno = detectar_turno(p_norm)
        if not turno or turno == "ead":
            return ok("Não reconheci o turno. Informe: Matutino, Vespertino ou Noturno.", sid)
        if turno == ctx.get("turno_atual"):
            return ok("O turno desejado é igual ao turno atual. Informe um turno diferente.", sid)
        ctx["turno_desejado"] = turno
        session["state"] = "trhorario.confirm"
        return ok(
            f"Confirmando sua solicitação:\n"
            f"• RA: {ctx.get('ra')}\n"
            f"• Turno atual: {ctx.get('turno_atual','').capitalize()}\n"
            f"• Turno desejado: {turno.capitalize()}\n\n"
            "Posso registrar? (sim/não)", sid
        )

    if state == "trhorario.confirm":
        if is_affirmative(p_norm):
            prazo = dados.get("transferencia_horario", {}).get("prazo", "consulte a secretaria")
            session["state"] = "idle"
            session["context"] = {}
            return ok(
                f"✅ Solicitação de transferência de horário registrada!\n\n"
                "Próximos passos:\n"
                "1. Compareça à secretaria presencial com documento de identificação\n"
                "2. Aguarde análise da coordenação (até 5 dias úteis)\n"
                "3. Acompanhe o resultado pelo portal acadêmico\n\n"
                f"⚠️ Atenção: {prazo}. Sujeito à disponibilidade de vagas.\n\n"
                "Posso ajudar em mais alguma coisa?", sid
            )
        if is_negative(p_norm):
            session["state"] = "idle"
            session["context"] = {}
            return ok("Solicitação cancelada. Fico à disposição se precisar!", sid)
        return ok("Os dados estão corretos? (sim/não)", sid)

    # =========================================================================
    # FLUXO: ESTÁGIO
    # =========================================================================
    if state == "estagio.await_tipo":
        if any(k in p_norm for k in ["obrigatorio", "obrigatório", "curricular"]):
            info = dados.get("estagio", {}).get("obrigatorio", {})
            docs = info.get("documentos", [])
            lines = ["📋 Estágio Obrigatório (Curricular):\n"]
            lines.append(f"• Carga horária: {info.get('carga_horaria','consulte a coordenação')}")
            lines.append(f"• Início: {info.get('quando_iniciar','consulte a grade curricular')}")
            lines.append("\nDocumentos necessários:")
            for d in docs:
                lines.append(f"  – {d}")
            lines.append(f"\n📧 Entrega: {dados.get('estagio',{}).get('contato','secretaria')}")
            lines.append("\nPosso ajudar com mais alguma coisa?")
            session["state"] = "idle"
            return ok(format_blocks(None, lines), sid)

        if any(k in p_norm for k in ["nao obrigatorio", "não obrigatório", "nao obrigatório", "extracurricular", "voluntario"]):
            info = dados.get("estagio", {}).get("nao_obrigatorio", {})
            reqs = info.get("requisitos", [])
            docs = info.get("documentos", [])
            lines = ["📋 Estágio Não Obrigatório (Extracurricular):\n"]
            lines.append("Requisitos:")
            for r in reqs:
                lines.append(f"  – {r}")
            lines.append("\nDocumentos necessários:")
            for d in docs:
                lines.append(f"  – {d}")
            lines.append(f"\n📧 Contato: {dados.get('estagio',{}).get('contato','secretaria')}")
            lines.append("\nPosso ajudar com mais alguma coisa?")
            session["state"] = "idle"
            return ok(format_blocks(None, lines), sid)

        return ok(
            "Você quer informações sobre:\n\n"
            "1️⃣ Estágio Obrigatório (curricular)\n"
            "2️⃣ Estágio Não Obrigatório (extracurricular)\n\n"
            "Digite 1 ou o nome do tipo de estágio.", sid
        )

    # Handle "1" and "2" shortcuts inside estagio flow
    if state == "estagio.await_tipo" or state == "idle":
        pass  # handled below as intent, not here

    # =========================================================================
    # DETECÇÃO DE INTENÇÃO (estado idle)
    # =========================================================================

    # ── Matrícula ──
    if any(k in p_norm for k in ["matricula", "rematricula"]):
        procedimento = dados.get("matrícula", {}).get("procedimento", [])
        lines = ["Veja como funciona o processo de matrícula:\n"]
        for i, passo in enumerate(procedimento, 1):
            lines.append(f"{i}. {passo}")
        lines.append("\nDeseja que eu te ajude a iniciar o processo agora? (sim/não)")
        session["state"] = "matricula.await_confirm"
        return ok(format_blocks(None, lines), sid)

    # ── Trancamento ──
    if any(k in p_norm for k in ["trancamento", "trancar"]):
        info = dados.get("trancamento", {})
        lines = ["Trancamento de vínculo acadêmico:\n"]
        if desc := info.get("descricao"):
            lines.append(desc)
        if regras := info.get("regras"):
            lines.append("\nRegras importantes:")
            for r in regras:
                lines.append(f"• {r}")
        lines.append("\nDeseja que eu inicie o pedido de trancamento? (sim/não)")
        session["state"] = "trancamento.await_confirm"
        return ok(format_blocks(None, lines), sid)

    # ── Documentos ──
    if any(k in p_norm for k in ["documento", "historico", "declaracao", "comprovante"]):
        info  = dados.get("documentos", {})
        nomes = {
            "historico_escolar":    "Histórico Escolar",
            "declaracao_matricula": "Declaração de Matrícula",
            "ementa_disciplina":    "Ementa de Disciplina",
        }
        lines = ["Documentos disponíveis e como obtê-los:\n"]
        for key, val in info.items():
            nome  = nomes.get(key, key.replace("_", " ").capitalize())
            canal = val.get("canal", "") if isinstance(val, dict) else ""
            prazo = val.get("prazo", "") if isinstance(val, dict) else ""
            lines.append(f"📄 {nome}")
            lines.append(f"   Canal: {canal}")
            lines.append(f"   Prazo: {prazo}\n")
        lines.append("Deseja instruções para solicitar? (sim/não)")
        session["state"] = "documentos.await_confirm"
        return ok(format_blocks(None, lines), sid)

    # ── Transferência de horário ──
    if any(k in p_norm for k in ["transferencia", "transferir", "mudar horario", "mudar turno", "trocar turno", "trocar horario"]):
        info  = dados.get("transferencia_horario", {})
        regras = info.get("regras", [])
        lines = ["Transferência de turno/horário de aula:\n"]
        if desc := info.get("descricao"):
            lines.append(desc)
        if regras:
            lines.append("\nRegras:")
            for r in regras:
                lines.append(f"• {r}")
        lines.append("\nDeseja iniciar sua solicitação? (sim/não)")
        session["state"] = "trhorario.await_confirm"
        return ok(format_blocks(None, lines), sid)

    # ── Calendário acadêmico ──
    if any(k in p_norm for k in ["calendario", "calend", "datas", "data importante", "semestre"]):
        cal  = dados.get("calendario_academico", {})
        datas = cal.get("datas", {})
        obs   = cal.get("observacao", "")
        semestre = cal.get("semestre_atual", "")
        lines = [f"📅 Calendário Acadêmico — {semestre}\n"]
        for evento, data in datas.items():
            lines.append(f"• {evento}: {data}")
        if obs:
            lines.append(f"\n⚠️ {obs}")
        lines.append("\nPosso ajudar com mais alguma coisa?")
        return ok(format_blocks(None, lines), sid)

    # ── Estágio ──
    if any(k in p_norm for k in ["estagio", "estágio", "estagiar", "estagios"]):
        info = dados.get("estagio", {})
        lines = [f"🎓 Estágios — {info.get('descricao','')}\n"]
        lines.append("Você quer informações sobre:\n")
        lines.append("1️⃣ Estágio Obrigatório (curricular)")
        lines.append("2️⃣ Estágio Não Obrigatório (extracurricular)\n")
        lines.append("Digite 1, 2, ou o nome do tipo.")
        session["state"] = "estagio.await_tipo"
        return ok(format_blocks(None, lines), sid)

    # ── Disciplinas e professores ──
    disciplinas = dados.get("informacoes", {}).get("disciplinas", {})

    # Listagem geral de todas as disciplinas
    if any(k in p_norm for k in ["disciplinas", "grade", "materias", "matérias", "todas as disciplinas"]):
        lines = ["📚 Disciplinas do semestre atual:\n"]
        for cod, info in disciplinas.items():
            materia  = info.get("materia", cod)
            professor = info.get("professor", "A informar")
            lines.append(f"• [{cod}] {materia}")
            lines.append(f"       Prof(a): {professor}\n")
        lines.append("Para saber mais sobre uma disciplina específica, informe o código ou o nome.")
        return ok(format_blocks(None, lines), sid)

    # Busca por disciplina específica
    resultado = buscar_disciplina(p_norm, disciplinas)
    if resultado:
        codigo, info = resultado
        materia  = info.get("materia", codigo)
        professor = info.get("professor", "Não informado")
        horario   = info.get("horario", "-- A PREENCHER --")
        sala      = info.get("sala",    "-- A PREENCHER --")

        if any(k in p_norm for k in ["professor", "quem", "ministra", "leciona", "responsavel"]):
            return ok(f"👨‍🏫 {materia} ({codigo})\nProfessor(a): {professor}", sid)

        if any(k in p_norm for k in ["horario", "hora", "aula"]):
            return ok(f"🕐 Horário de {materia} ({codigo}): {horario}", sid)

        if any(k in p_norm for k in ["ementa", "conteudo", "objetivo"]):
            return ok(
                f"A ementa de {materia} deve ser solicitada na secretaria "
                "presencial ou por e-mail institucional. Prazo: até 5 dias úteis.", sid
            )

        # Resposta geral
        return ok(
            f"📚 {materia}\n"
            f"Código: {codigo}\n"
            f"Professor(a): {professor}\n"
            f"Horário: {horario}\n"
            f"Sala: {sala}\n\n"
            "Para obter a ementa, acesse a secretaria ou envie e-mail institucional.", sid
        )

    # ── Fallback Anthropic ──
    contexto_str = json.dumps(dados, ensure_ascii=False, indent=2)
    resposta_ia  = await consultar_claude(contexto_str, request.pergunta)
    if resposta_ia:
        return ok(resposta_ia, sid, "claude_ai")

    # ── Fallback local ──
    return ok(
        "Não identifiquei sua solicitação. Posso ajudar com:\n\n"
        "• Matrícula e rematrícula\n"
        "• Trancamento de curso\n"
        "• Documentos (histórico, declarações)\n"
        "• Transferência de horário\n"
        "• Calendário acadêmico\n"
        "• Orientação sobre estágio\n"
        "• Disciplinas e professores\n\n"
        "O que você precisa?", sid, "fallback_local"
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "4.0.0"}
