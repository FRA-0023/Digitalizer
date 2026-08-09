#!/usr/bin/env python3
"""
Arricchimento Accademico degli Appunti  —  v1.1 (Gemini API)
════════════════════════════════════════════════════════════
Legge le pagine già presenti in un database Notion (es. quelle create
dalla pipeline OCR) e le invia a Google Gemini con un prompt da
"Professore Universitario" che:
  1. Corregge sintassi, abbreviazioni e refusi
  2. Ristruttura il testo in Markdown accademico (H1/H2/H3, grassetti, elenchi)
  3. Arricchisce i concetti abbozzati con note del professore (blockquote)

Le pagine troppo lunghe per il modello vengono divise in blocchi (ai
confini delle sezioni ##) e processate in sequenza, poi riassemblate.

Tutto è ripristinabile: stato salvato su disco, backup del testo
originale prima di ogni modifica.

NOTA: usa l'API cloud di Google Gemini per velocità. In futuro è
prevista una versione locale via Ollama.
════════════════════════════════════════════════════════════
"""

import os
import re
import json
import time
import logging
import unicodedata
from pathlib import Path

from google import genai    # pip install google-genai
import requests        # pip install requests
from dotenv import load_dotenv   # pip install python-dotenv

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, MofNCompleteColumn, TimeRemainingColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.rule  import Rule
from rich       import box

# ════════════════════════════════════════════════════════════
# 1. CONFIGURAZIONE
# ════════════════════════════════════════════════════════════
load_dotenv()
console = Console()

NOTION_TOKEN        = os.getenv("NOTION_TOKEN")
NOTION_ROOT_PAGE_ID  = os.getenv("NOTION_ROOT_PAGE_ID")
NOTION_API_BASE      = "https://api.notion.com/v1"

# Modello AI per l'arricchimento — Google Gemini API (veloce, cloud).
# In futuro si potrà tornare a un modello locale via Ollama.
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
MODEL_PROFESSORE  = os.getenv("MODEL_PROFESSORE", "gemini-3.5-flash-lite")
# Oltre questa soglia di caratteri, una pagina viene divisa in blocchi
# prima di essere inviata al modello (per non saturare il contesto).
CHUNK_CHARS = 4000

DIR_BACKUP            = Path("backup_pre_arricchimento")
DIR_CACHE_ARRICCHITO  = Path("cache_arricchito")
DIR_BACKUP.mkdir(exist_ok=True)
DIR_CACHE_ARRICCHITO.mkdir(exist_ok=True)

if not NOTION_TOKEN or not NOTION_ROOT_PAGE_ID:
    console.print(Panel(
        "[bold red]ERRORE CRITICO[/bold red]\n"
        "Configura [cyan]NOTION_TOKEN[/cyan] e [cyan]NOTION_ROOT_PAGE_ID[/cyan] "
        "nel file [bold].env[/bold]",
        border_style="red",
    ))
    raise SystemExit(1)

if not GEMINI_API_KEY:
    console.print(Panel(
        "[bold red]ERRORE CRITICO[/bold red]\n"
        "Configura [cyan]GEMINI_API_KEY[/cyan] nel file [bold].env[/bold]\n"
        "Ottieni una chiave gratuita su: [cyan]https://aistudio.google.com/apikey[/cyan]",
        border_style="red",
    ))
    raise SystemExit(1)

_gemini_client = genai.Client(api_key=GEMINI_API_KEY)

logger = logging.getLogger("Enrich")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler("enrich.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_fh)

DEFAULT_ANNOTATIONS = {
    "bold": False, "italic": False, "strikethrough": False,
    "underline": False, "code": False, "color": "default",
}


def retry(retries: int = 3, delay: float = 2.0):
    """Decoratore: riprova automaticamente in caso di errore di rete."""
    def dec(fn):
        def wrapper(*a, **kw):
            for attempt in range(retries):
                try:
                    return fn(*a, **kw)
                except Exception as exc:
                    if attempt == retries - 1:
                        raise
                    console.print(
                        f"  [yellow]⚠  Retry {attempt+1}/{retries} "
                        f"({type(exc).__name__}: {exc})[/yellow]"
                    )
                    time.sleep(delay)
        return wrapper
    return dec


# ════════════════════════════════════════════════════════════
# 2. CONVERSIONE MARKDOWN → BLOCCHI NOTION
# ════════════════════════════════════════════════════════════
def _parse_inline(text: str) -> list[dict]:
    """
    Divide una riga in segmenti rich_text Notion, gestendo **grassetto**.
    Esempio: "Il **PIL** misura..." → [testo normale, "PIL" in grassetto, testo normale]
    """
    if not text:
        return [{"type": "text", "text": {"content": ""},
                 "annotations": dict(DEFAULT_ANNOTATIONS)}]

    segments = []
    parts = re.split(r"(\*\*.+?\*\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            content = part[2:-2][:2000]
            ann = dict(DEFAULT_ANNOTATIONS)
            ann["bold"] = True
            segments.append({"type": "text", "text": {"content": content},
                             "annotations": ann})
        else:
            segments.append({"type": "text", "text": {"content": part[:2000]},
                             "annotations": dict(DEFAULT_ANNOTATIONS)})
    return segments if segments else [
        {"type": "text", "text": {"content": ""}, "annotations": dict(DEFAULT_ANNOTATIONS)}
    ]


def markdown_to_blocks(text: str) -> list[dict]:
    """
    Converte markdown (con la sintassi del prompt del professore) in blocchi Notion:
      # Titolo      → heading_1
      ## Sezione    → heading_2
      ### Concetto  → heading_3
      > Nota        → quote
      - item        → bulleted_list_item
      1. item       → numbered_list_item
      | a | b |     → table (con table_row annidate)
      **grassetto** → rich_text con annotations.bold
    """
    blocks = []
    prev_empty = False

    lines = text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        raw_line = lines[i]
        stripped = raw_line.strip()

        # ── Tabella Markdown ──
        # Richiede: riga | ... | seguita da riga separatore |---|---|
        if stripped.startswith("|") and i + 1 < n:
            sep_line = lines[i + 1].strip()
            if re.match(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$", sep_line):
                header_cells = [c.strip() for c in stripped.strip("|").split("|")]
                n_cols = len(header_cells)

                righe_tabella = [header_cells]
                j = i + 2
                while j < n and lines[j].strip().startswith("|"):
                    row_cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    row_cells = (row_cells + [""] * n_cols)[:n_cols]
                    righe_tabella.append(row_cells)
                    j += 1

                table_row_blocks = [
                    {
                        "object": "block",
                        "type": "table_row",
                        "table_row": {"cells": [_parse_inline(cell) for cell in row]},
                    }
                    for row in righe_tabella
                ]

                blocks.append({
                    "object": "block",
                    "type": "table",
                    "table": {
                        "table_width": n_cols,
                        "has_column_header": True,
                        "has_row_header": False,
                        "children": table_row_blocks,
                    },
                })

                prev_empty = False
                i = j
                continue

        if stripped.startswith("# "):
            content = stripped[2:].strip()
            if content:
                blocks.append({"object": "block", "type": "heading_1",
                               "heading_1": {"rich_text": _parse_inline(content)}})
            prev_empty = False

        elif stripped.startswith("## "):
            content = stripped[3:].strip()
            if content:
                blocks.append({"object": "block", "type": "heading_2",
                               "heading_2": {"rich_text": _parse_inline(content)}})
            prev_empty = False

        elif stripped.startswith("### "):
            content = stripped[4:].strip()
            if content:
                blocks.append({"object": "block", "type": "heading_3",
                               "heading_3": {"rich_text": _parse_inline(content)}})
            prev_empty = False

        elif stripped.startswith("> "):
            content = stripped[2:].strip()
            if content:
                blocks.append({"object": "block", "type": "quote",
                               "quote": {"rich_text": _parse_inline(content)}})
            prev_empty = False

        elif re.match(r"^[-*]\s+", stripped):
            content = re.sub(r"^[-*]\s+", "", stripped)
            if content:
                blocks.append({"object": "block", "type": "bulleted_list_item",
                               "bulleted_list_item": {"rich_text": _parse_inline(content)}})
            prev_empty = False

        elif re.match(r"^\d+\.\s+", stripped):
            content = re.sub(r"^\d+\.\s+", "", stripped)
            if content:
                blocks.append({"object": "block", "type": "numbered_list_item",
                               "numbered_list_item": {"rich_text": _parse_inline(content)}})
            prev_empty = False

        elif not stripped:
            if not prev_empty:
                blocks.append({"object": "block", "type": "paragraph",
                               "paragraph": {"rich_text": [
                                   {"type": "text", "text": {"content": " "}}
                               ]}})
            prev_empty = True

        else:
            blocks.append({"object": "block", "type": "paragraph",
                           "paragraph": {"rich_text": _parse_inline(stripped)}})
            prev_empty = False

        i += 1

    return blocks


# ════════════════════════════════════════════════════════════
# 3. NOTION API
# ════════════════════════════════════════════════════════════
class NotionAPI:
    def __init__(self, token: str):
        self._h = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    def _check(self, r):
        if not r.ok:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:500]
            raise requests.HTTPError(f"{r.status_code} {r.reason} — {detail}", response=r)
        return r.json()

    @retry()
    def _get(self, ep, params=None):
        r = requests.get(f"{NOTION_API_BASE}/{ep}", headers=self._h, params=params, timeout=30)
        return self._check(r)

    @retry()
    def _post(self, ep, payload):
        r = requests.post(f"{NOTION_API_BASE}/{ep}", headers=self._h, json=payload, timeout=30)
        return self._check(r)

    @retry()
    def _patch(self, ep, payload):
        r = requests.patch(f"{NOTION_API_BASE}/{ep}", headers=self._h, json=payload, timeout=30)
        return self._check(r)

    @retry()
    def _delete(self, ep):
        r = requests.delete(f"{NOTION_API_BASE}/{ep}", headers=self._h, timeout=30)
        return self._check(r)

    def fetch_blocks(self, parent_id: str) -> list:
        results, params = [], {"page_size": 100}
        while True:
            data = self._get(f"blocks/{parent_id}/children", params)
            results.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            params["start_cursor"] = data["next_cursor"]
        return results

    def query_db(self, db_id: str, filters: dict | None = None) -> list:
        results, payload = [], {"page_size": 100}
        if filters:
            payload["filter"] = filters
        while True:
            data = self._post(f"databases/{db_id}/query", payload)
            results.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            payload["start_cursor"] = data["next_cursor"]
        return results

    def list_active_pages(self, db_id: str) -> list[dict]:
        """Pagine non archiviate del database, con id e titolo."""
        pages = self.query_db(db_id)
        out = []
        for p in pages:
            title_prop = p.get("properties", {}).get("Name", {}).get("title", [])
            title = title_prop[0]["plain_text"] if title_prop else "(senza titolo)"
            out.append({"id": p["id"], "title": title})
        return out

    def fetch_page_text(self, page_id: str) -> str:
        """Estrae il testo come markdown (heading/bullet/quote/paragraph)."""
        blocks = self.fetch_blocks(page_id)
        lines = []
        for b in blocks:
            btype = b.get("type", "")
            data  = b.get(btype, {})
            rich  = data.get("rich_text", [])
            text  = "".join(r.get("plain_text", "") for r in rich)
            if btype == "heading_1":
                lines.append(f"# {text}")
            elif btype == "heading_2":
                lines.append(f"## {text}")
            elif btype == "heading_3":
                lines.append(f"### {text}")
            elif btype == "bulleted_list_item":
                lines.append(f"- {text}")
            elif btype == "numbered_list_item":
                lines.append(f"1. {text}")
            elif btype == "quote":
                lines.append(f"> {text}")
            else:
                lines.append(text)
        return "\n".join(lines)

    def clear_blocks(self, page_id: str):
        """Elimina (archivia) tutti i blocchi figli di una pagina."""
        blocks = self.fetch_blocks(page_id)
        for b in blocks:
            try:
                self._delete(f"blocks/{b['id']}")
            except Exception as exc:
                logger.warning(f"Impossibile eliminare blocco {b['id']}: {exc}")

    def append_markdown(self, page_id: str, markdown_text: str):
        """Converte il markdown in blocchi Notion e li aggiunge alla pagina, a gruppi di 20."""
        blocks = markdown_to_blocks(markdown_text)
        CHUNK = 20
        for i in range(0, len(blocks), CHUNK):
            chunk = blocks[i:i + CHUNK]
            self._patch(f"blocks/{page_id}/children", {"children": chunk})
            time.sleep(0.3)
        return len(blocks)


# ════════════════════════════════════════════════════════════
# 4. AI ENGINE — Prompt del Professore
# ════════════════════════════════════════════════════════════
PROMPT_TEMPLATE = """# CHI SEI
Sei uno studente universitario estremamente bravo (materie economiche, quantitative, data science) che sta riscrivendo i propri appunti di lezione per ripassare meglio. 
Non stai scrivendo un libro o una dispensa ufficiale: stai sistemando i TUOI appunti per capirli e studiarli bene.

# COSA HAI DAVANTI
Appunti grezzi presi di fretta a lezione: ci sono abbreviazioni da tastiera (es. "-d" per "-->"), refusi (es. "wuoghi" invece di "luoghi") e frasi lasciate a metà.

# COSA DEVI FARE

1. Pulizia
- Sistema abbreviazioni, refusi ed errori di battitura.
- Completa le frasi spezzate così come le finiresti tu, senza cambiare il senso di quello che è scritto.

2. Struttura
{heading_rule}
- ## per le sezioni principali del discorso
- ### per i concetti/definizioni specifiche dentro una sezione
- elenchi puntati per le cose elencate
- **grassetto** sui termini tecnici importanti
- Emoji contestuale: metti UNA emoji pertinente all'inizio di ogni heading (## e ###) per segnalare subito l'argomento (es. "## 📊 Analisi di mercato", "### ⚖️ Politica fiscale"). Niente emoji dentro il testo normale, solo negli heading.

3. Completa dove serve
- Se un concetto è solo accennato o sembra mancare un pezzo, aggiungilo tu, come faresti se lo stessi spiegando a te stesso.
- Integra l'aggiunta nel discorso normale, senza etichette o riquadri tipo "nota": deve sembrare parte naturale degli appunti, non un commento esterno.

# COME DEVE SUONARE IL TESTO — IMPORTANTE
Scrivi come scriveresti tu per ripassare, NON come un libro di testo o un'intelligenza artificiale:
- Frasi dirette, brevi quando possono esserlo.
- Evita frasi fatte da tema scolastico tipo "è importante sottolineare che", "risulta evidente come", "in conclusione possiamo affermare".
- Niente introduzioni tipo "Ecco gli appunti rielaborati:" — vai dritto al contenuto.
- Tono colloquiale dove aiuta a capire meglio, ma resta preciso su numeri, formule e passaggi logici: lì niente semplificazioni che cambiano il senso.

# REGOLE DI FORMATTAZIONE
- Vai a capo dopo ogni frase completata, per rendere il testo scorrevole da leggere.
    IMPORTANTE: usa un semplice ritorno a capo (newline), MAI aggiungere spazi finali prima di andare a capo. La riga deve terminare esattamente con l'ultimo carattere della frase, nessuno spazio dopo.
- Negli elenchi, scrivi tutto l'elemento sulla stessa riga del trattino/numero, senza andare a capo a metà.
- Mantieni la rigorosità se ci sono formule, dati o passaggi logici.
- Formule matematiche: tienile semplici e leggibili come testo semplice.
    - Formula inline: racchiudi in singolo `$` (es. $E=mc^2$). Preferisci notazione piatta come $(G_t - T_t)/Y_t$ invece di comandi LaTeX annidati.
    - EVITA `\\frac{{}}{{}}`, `\\sum`, `\\int` e altri comandi LaTeX complessi dentro formule inline `$...$`: spesso non si renderizzano e restano visibili come codice rotto.
    - Usa `\\frac{{}}{{}}` e comandi complessi SOLO in formule display su riga propria con `$$...$$` (es. $$\\frac{{G_t - T_t}}{{Y_t}}$$).
    - Mai usare blocchi di codice per le formule.
- Tabelle: se servono, usa sintassi Markdown standard con pipe `|`, una riga di intestazione, una riga separatore `|---|---|`, e lo stesso numero di colonne in ogni riga. Non lasciare celle con contenuto multi-riga.

# APPUNTI DA SISTEMARE
{titolo_riga}

{testo}

# OUTPUT
Scrivi solo gli appunti sistemati in markdown, senza nient'altro prima o dopo."""

class AIEngine:

    @staticmethod
    def _chiama_gemini(prompt: str, tentativi: int = 4) -> str:
        """
        Chiama Gemini con retry esponenziale per rate limit temporanei (RPM).
        Se la quota esaurita è giornaliera (RPD), fallisce subito senza retry:
        nessuna attesa di secondi risolve un limite che si resetta a mezzanotte.
        """
        delay = 3
        for tentativo in range(tentativi):
            try:
                res = _gemini_client.models.generate_content(
                    model=MODEL_PROFESSORE,
                    contents=prompt,
                )
                return (res.text or "").strip()
            except Exception as exc:
                msg = str(exc)
                is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                is_daily_quota = "PerDay" in msg or "GenerateRequestsPerDay" in msg

                if is_daily_quota:
                    console.print(
                        "  [red]✗  Quota giornaliera Gemini esaurita "
                        "(free tier: 20 richieste/giorno per questo modello).\n"
                        "     Aspetta il reset o passa a un piano a pagamento.[/red]"
                    )
                    raise

                if tentativo == tentativi - 1:
                    raise
                attesa = delay * (2 ** tentativo) if is_rate_limit else delay
                console.print(
                    f"  [yellow]⚠  Retry {tentativo+1}/{tentativi} "
                    f"({'rate limit' if is_rate_limit else type(exc).__name__}) "
                    f"— attendo {attesa}s…[/yellow]"
                )
                time.sleep(attesa)
        return ""

    @staticmethod
    def enrich(testo: str, titolo: str, primo_blocco: bool = True) -> str:
        """
        Invia un blocco di testo a Gemini con il prompt del professore.
        primo_blocco=False evita di ripetere il titolo H1 nei blocchi successivi
        di una stessa pagina troppo lunga per essere processata in un colpo solo.
        """
        if primo_blocco:
            heading_rule = f'- Usa un singolo # (H1) per il titolo principale: "{titolo}"'
            titolo_riga  = f'Titolo della lezione: "{titolo}"'
        else:
            heading_rule = "- NON ripetere il titolo principale: continua direttamente con ## per le sezioni"
            titolo_riga  = "(continuazione della stessa lezione, non ripetere il titolo)"

        prompt = PROMPT_TEMPLATE.format(
            heading_rule=heading_rule, titolo_riga=titolo_riga, testo=testo
        )
        return AIEngine._chiama_gemini(prompt)

    @staticmethod
    def verifica_modello() -> bool:
        try:
            res = _gemini_client.models.generate_content(
                model=MODEL_PROFESSORE,
                contents="Rispondi solo con: ok",
            )
            return bool((res.text or "").strip())
        except Exception as exc:
            logger.error(f"Verifica modello fallita: {exc}")
            return False


def split_in_chunks(testo: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """
    Divide un testo lungo in blocchi più piccoli, preferendo i confini
    delle sezioni ## (così ogni blocco resta semanticamente coerente).
    Se una singola sezione supera comunque il limite, la spezza per paragrafi.
    """
    if len(testo) <= max_chars:
        return [testo]

    sezioni = re.split(r"(?=^## )", testo, flags=re.MULTILINE)
    sezioni = [s for s in sezioni if s.strip()]

    chunks, current = [], ""
    for sec in sezioni:
        if len(current) + len(sec) <= max_chars:
            current += sec
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(sec) <= max_chars:
            current = sec
        else:
            # Sezione singola troppo grande: spezza per paragrafi
            sub = ""
            for para in sec.split("\n\n"):
                if len(sub) + len(para) <= max_chars:
                    sub += para + "\n\n"
                else:
                    if sub:
                        chunks.append(sub)
                    sub = para + "\n\n"
            if sub:
                chunks.append(sub)
    if current:
        chunks.append(current)

    return chunks if chunks else [testo]


# ════════════════════════════════════════════════════════════
# 5. RICERCA DATABASE (stessa logica della pipeline OCR)
# ════════════════════════════════════════════════════════════
def _norm(text: str) -> str:
    for ch in ("\u2019", "\u2018", "\u02BC", "\u0060", "\u00B4"):
        text = text.replace(ch, "'")
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ASCII", "ignore").decode("utf-8")
        .lower().strip()
    )


def trova_database(api: NotionAPI, subject: str) -> str:
    console.print(f"\n[cyan]📡 Ricerca corso [bold]'{subject}'[/bold] su Notion…[/cyan]")
    courses = api.query_db(NOTION_ROOT_PAGE_ID)

    match = next(
        (c for c in courses
         if _norm(subject) in _norm(
             c.get("properties", {}).get("Name", {})
              .get("title", [{}])[0].get("plain_text", "")
         )), None,
    )
    if not match:
        raise RuntimeError(f"Nessun corso trovato per '{subject}'.")

    nome_corso = match["properties"]["Name"]["title"][0]["plain_text"]
    console.print(f"  [green]✓ Corso trovato:[/green] {nome_corso}")

    keywords = {"note", "appunti", "notes"}
    CONTENITORI = {"child_page", "column_list", "column"}

    def cerca_ricorsivo(parent_id: str, profondita: int = 0, max_profondita: int = 6):
        if profondita > max_profondita:
            return None
        blocks = api.fetch_blocks(parent_id)

        # Match diretto a questo livello: child_database pertinente
        for b in blocks:
            if b.get("type") == "child_database" and any(
                k in _norm(b.get("child_database", {}).get("title", ""))
                for k in keywords
            ):
                return b

        # Nessun match diretto: scendi nei contenitori strutturali
        # (child_page, ma anche column_list/column usate per i layout a bottoni)
        for b in blocks:
            if b.get("type") in CONTENITORI:
                trovato = cerca_ricorsivo(b["id"], profondita + 1, max_profondita)
                if trovato:
                    return trovato
        return None

    db = cerca_ricorsivo(match["id"])

    if not db:
        raise RuntimeError("Database 'Note' / 'Appunti' non trovato nel corso.")
    return db["id"]

# ════════════════════════════════════════════════════════════
# 6. PROCESSO DI ARRICCHIMENTO
# ════════════════════════════════════════════════════════════
def safe_filename(text: str) -> str:
    return re.sub(r"[^\w\-]+", "_", text)[:80]


def arricchisci_database(api: NotionAPI, database_id: str, db_label: str):
    stato_path = Path(f"stato_arricchimento_{safe_filename(db_label)}.json")
    stato = {"completate": []}
    if stato_path.exists():
        try:
            stato = json.loads(stato_path.read_text(encoding="utf-8"))
            console.print(
                f"[cyan]♻  Stato precedente trovato: "
                f"{len(stato['completate'])} pagine già arricchite[/cyan]"
            )
        except Exception:
            pass

    def salva_stato():
        stato_path.write_text(json.dumps(stato, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Verifica modello ──
    console.print(f"\n[dim]Verifica modello {MODEL_PROFESSORE}…[/dim]")
    if not AIEngine.verifica_modello():
        console.print(
            f"[red]✗  Modello {MODEL_PROFESSORE} non disponibile.\n"
            f"   Verifica che GEMINI_API_KEY sia valida e che il modello esista.[/red]"
        )
        return
    console.print(f"  [green]✓  Modello operativo[/green]")

    # ── Elenco pagine ──
    console.print("\n[cyan]📂 Caricamento elenco pagine dal database…[/cyan]")
    pagine = api.list_active_pages(database_id)
    # Salta eventuali bozze batch non ancora archiviate
    pagine = [p for p in pagine if " - Parte " not in p["title"]]

    if not pagine:
        console.print("[yellow]Nessuna pagina trovata nel database.[/yellow]")
        return

    console.print(f"  [green]✓  {len(pagine)} pagine trovate[/green]")

    tbl = Table(title="📋 Pagine nel database", box=box.ROUNDED,
                header_style="bold cyan", show_lines=True)
    tbl.add_column("N.", justify="right", width=4)
    tbl.add_column("Titolo", min_width=40)
    tbl.add_column("Stato", justify="center", width=14)
    for i, p in enumerate(pagine, 1):
        stato_p = "[green]✓ Fatta[/green]" if p["id"] in stato["completate"] else "[dim]Da fare[/dim]"
        tbl.add_row(str(i), p["title"], stato_p)
    console.print(tbl)

    da_processare = [p for p in pagine if p["id"] not in stato["completate"]]
    if not da_processare:
        console.print("[green]✓  Tutte le pagine sono già state arricchite.[/green]")
        return

    console.print(f"\n[cyan]✎  Arricchimento di {len(da_processare)} pagine…[/cyan]\n")

    successi = errori = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
                  console=console, transient=False) as prog:
        task = prog.add_task("[cyan]Pagine", total=len(da_processare))

        for p in da_processare:
            page_id, titolo = p["id"], p["title"]
            prog.print(Rule(f"[bold blue]{titolo}[/bold blue]", style="blue"))

            try:
                # ── STEP 1/5: Recupera testo originale ──
                prog.print("  [dim][1/5][/dim] Lettura testo originale da Notion…")
                testo_originale = api.fetch_page_text(page_id)
                if not testo_originale.strip():
                    prog.print("  [dim]⚠  Pagina vuota, saltata[/dim]")
                    prog.advance(task, 1)
                    continue
                prog.print(f"  [dim]      → {len(testo_originale)} caratteri letti[/dim]")

                # ── STEP 2/5: Backup locale prima di qualsiasi modifica ──
                backup_file = DIR_BACKUP / f"{safe_filename(titolo)}.md"
                if not backup_file.exists():
                    backup_file.write_text(testo_originale, encoding="utf-8")
                    prog.print(f"  [dim][2/5][/dim] Backup salvato → {backup_file.name}")
                else:
                    prog.print(f"  [dim][2/5][/dim] Backup già presente, non sovrascritto")

                # ── STEP 3/5: Arricchimento (a blocchi se necessario) ──
                cache_file = DIR_CACHE_ARRICCHITO / f"{safe_filename(titolo)}.md"
                if cache_file.exists():
                    testo_arricchito = cache_file.read_text(encoding="utf-8")
                    prog.print(f"  [dim][3/5][/dim] Versione arricchita già in cache, riuso")
                else:
                    chunks = split_in_chunks(testo_originale)
                    if len(chunks) > 1:
                        prog.print(f"  [dim][3/5][/dim] Pagina lunga → divisa in {len(chunks)} blocchi")
                    else:
                        prog.print(f"  [dim][3/5][/dim] Invio a Gemini…")

                    risultati = []
                    for i, chunk in enumerate(chunks):
                        if len(chunks) > 1:
                            prog.print(f"        ↳ blocco {i+1}/{len(chunks)}…")
                        out = AIEngine.enrich(chunk, titolo, primo_blocco=(i == 0))
                        risultati.append(out)
                        time.sleep(1)   # rispetta i rate limit del tier gratuito Gemini

                    testo_arricchito = "\n\n".join(risultati)

                    # ── Controllo qualità prima di salvare in cache ──
                    # Se Gemini ha restituito molto meno testo dell'originale,
                    # probabilmente la risposta è troncata o vuota: non la usiamo.
                    if len(testo_arricchito) < 0.3 * len(testo_originale):
                        raise ValueError(
                            f"Output sospetto: {len(testo_arricchito)} caratteri contro "
                            f"{len(testo_originale)} originali (possibile risposta troncata)"
                        )

                    cache_file.write_text(testo_arricchito, encoding="utf-8")
                    prog.print(f"  [dim]      → {len(testo_arricchito)} caratteri generati, salvati in cache[/dim]")

                # ── STEP 4/5: Sovrascrive la pagina su Notion ──
                prog.print("  [dim][4/5][/dim] Aggiornamento pagina su Notion…")
                api.clear_blocks(page_id)
                n_blocchi = api.append_markdown(page_id, testo_arricchito)
                prog.print(f"  [dim]      → {n_blocchi} blocchi caricati[/dim]")

                # ── STEP 5/5: Checkpoint ──
                successi += 1
                stato["completate"].append(page_id)
                salva_stato()
                prog.print(
                    f"  [dim][5/5][/dim] 💾 Checkpoint salvato "
                    f"({len(stato['completate'])}/{len(pagine)} pagine totali)"
                )
                prog.print(f"  [green]✓  Fatto: '{titolo}'[/green]")
                logger.info(f"Arricchita: {titolo}")

            except Exception as exc:
                prog.print(f"  [red]✗  Errore: {exc}[/red]")
                prog.print(f"  [yellow]   La pagina originale su Notion NON è stata toccata.[/yellow]")
                logger.error(f"Errore su '{titolo}': {exc}")
                errori += 1

            prog.advance(task, 1)

    console.print(Panel(
        f"[bold green]✅ Arricchimento completato[/bold green]\n\n"
        f"  Completate : [green]{successi}[/green]\n"
        f"  Errori     : [red]{errori}[/red]\n\n"
        f"[dim]Backup originali in: {DIR_BACKUP}/[/dim]",
        border_style="green",
    ))

    if errori == 0 and not da_processare[len(da_processare):]:
        # Tutte le pagine completate con successo: pulizia stato
        if all(p["id"] in stato["completate"] for p in pagine):
            stato_path.unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ════════════════════════════════════════════════════════════
def main():
    console.print(Panel(
        "[bold cyan]🎓 Arricchimento Accademico degli Appunti[/bold cyan]\n"
        "Trasforma appunti grezzi in dispense da Professore Universitario",
        subtitle="v1.0",
        border_style="cyan",
    ))

    subject = console.input("\n[bold] Materia del corso su Notion: [/bold]").strip()

    api = NotionAPI(NOTION_TOKEN)
    try:
        db_id = trova_database(api, subject)
    except Exception as exc:
        console.print(f"[red]✗  Errore navigazione Notion: {exc}[/red]")
        return

    console.rule()
    arricchisci_database(api, db_id, subject)

    console.print(Panel(
        "[bold green]🎉 Programma terminato![/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()