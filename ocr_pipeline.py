#!/usr/bin/env python3
"""
Pipeline OCR → Notion  –  v2.0
════════════════════════════════════════════════════════════
FASE 1  OCR visivo (GLM) + correzione maiuscolo (Qwen)
        Upload a Notion a gruppi di 10 pagine.
        Controllo duplicati prima di ogni caricamento.

FASE 2  Carica il testo corretto già presente in cache,
        genera la mappa semantica con Qwen, suddivide
        gli appunti per argomento (anche cross-batch),
        crea le pagine finali su Notion e archivia le bozze.
════════════════════════════════════════════════════════════
"""

import os
import re
import json
import time
import logging
import unicodedata
import tempfile
from pathlib import Path

import fitz          # PyMuPDF  — pip install pymupdf
import ollama        # pip install ollama
import requests      # pip install requests
from dotenv import load_dotenv  # pip install python-dotenv

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, MofNCompleteColumn, TimeRemainingColumn,
)
from rich.panel   import Panel
from rich.table   import Table
from rich.rule    import Rule
from rich         import box

# ──────────────────────────────────────────────────────────
# 1. CONFIGURAZIONE
# ──────────────────────────────────────────────────────────
load_dotenv()

console = Console()

NOTION_TOKEN       = os.getenv("NOTION_TOKEN")
NOTION_ROOT_PAGE_ID = os.getenv("NOTION_ROOT_PAGE_ID")
NOTION_API_BASE    = "https://api.notion.com/v1"
BATCH_SIZE         = 10          # pagine per gruppo nella Fase 1

# Modello usato per la segmentazione semantica (Fase 2).
# Con 4GB VRAM usa qwen2.5:3b. Con più VRAM puoi salire a 7b o 14b.
MODEL_SEMANTICA    = "qwen2.5:14b"

DIR_CACHE_GREZZO = Path("cache_ocr_grezzo")
DIR_CACHE_PULITO = Path("cache_ocr_revisionato")
DIR_CACHE_GREZZO.mkdir(exist_ok=True)
DIR_CACHE_PULITO.mkdir(exist_ok=True)

if not NOTION_TOKEN or not NOTION_ROOT_PAGE_ID:
    console.print(Panel(
        "[bold red]ERRORE CRITICO[/bold red]\n"
        "Configura [cyan]NOTION_TOKEN[/cyan] e "
        "[cyan]NOTION_ROOT_PAGE_ID[/cyan] nel file [bold].env[/bold]",
        border_style="red"
    ))
    raise SystemExit(1)

# Logging su file (la console è gestita da rich)
logger = logging.getLogger("Digitalizer")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler("pipeline.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(_fh)


# ──────────────────────────────────────────────────────────
# 2. NOTION API
# ──────────────────────────────────────────────────────────
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


class NotionAPI:
    def __init__(self, token: str):
        self._h = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    # ── Chiamate HTTP ────────────────────────
    @retry()
    def _get(self, ep: str, params=None):
        r = requests.get(f"{NOTION_API_BASE}/{ep}", headers=self._h,
                         params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    @retry()
    def _post(self, ep: str, payload: dict):
        r = requests.post(f"{NOTION_API_BASE}/{ep}", headers=self._h,
                          json=payload, timeout=30)
        if not r.ok:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:500]
            raise requests.HTTPError(
                f"{r.status_code} {r.reason} — {detail}", response=r
            )
        return r.json()

    @retry()
    def _patch(self, ep: str, payload: dict):
        r = requests.patch(f"{NOTION_API_BASE}/{ep}", headers=self._h,
                           json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    # ── Metodi Notion ────────────────────────
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

    def page_exists(self, db_id: str, title: str) -> bool:
        """Ritorna True se esiste già una pagina con questo titolo esatto."""
        results = self.query_db(db_id, {
            "property": "Name",
            "title": {"equals": title}
        })
        return len(results) > 0

    def get_pages_starting_with(self, db_id: str, prefix: str) -> list:
        """Tutte le pagine il cui titolo inizia con il prefisso dato."""
        return self.query_db(db_id, {
            "property": "Name",
            "title": {"starts_with": prefix}
        })

    def get_page_by_title(self, db_id: str, title: str) -> dict | None:
        """Ritorna la prima pagina con questo titolo esatto, o None."""
        results = self.query_db(db_id, {
            "property": "Name",
            "title": {"equals": title}
        })
        return results[0] if results else None

    def fetch_page_text(self, page_id: str) -> str:
        """
        Estrae tutto il testo leggibile da una pagina Notion.
        Concatena il testo di tutti i blocchi paragrafo in ordine.
        """
        blocks = self.fetch_blocks(page_id)
        lines = []
        for b in blocks:
            btype = b.get("type", "")
            rich  = b.get(btype, {}).get("rich_text", [])
            text  = "".join(r.get("plain_text", "") for r in rich)
            if text.strip():
                lines.append(text)
        return "\n".join(lines)

    def create_page(self, db_id: str, title: str, text: str,
                    lezione: int | None = None):
        blocks = self._text_to_blocks(text)
        props  = {"Name": {"title": [{"text": {"content": title}}]}}
        if lezione is not None:
            props["Lezione"] = {"number": lezione}

        CHUNK = 20   # Notion è più stabile con chunk piccoli
        resp    = self._post("pages", {
            "parent": {"database_id": db_id},
            "properties": props,
            "children": blocks[:CHUNK],
        })
        page_id = resp["id"].strip()
        for i in range(CHUNK, len(blocks), CHUNK):
            chunk = blocks[i:i+CHUNK]
            try:
                self._patch(f"blocks/{page_id}/children", {"children": chunk})
                time.sleep(0.3)
            except Exception as exc:
                logger.error(
                    f"Errore blocchi {i}–{i+len(chunk)} di '{title}':\n{exc}\n"
                    f"Blocco problematico: {chunk[0]}"
                )
                console.print(
                    f"  [yellow]⚠  Blocchi {i}–{i+len(chunk)} saltati "
                    f"(errore Notion): {str(exc)[:200]}[/yellow]"
                )

    @staticmethod
    def _text_to_blocks(text: str) -> list[dict]:
        """
        Converte testo markdown in blocchi Notion.
        Regole Notion API:
        - I paragrafi vuoti (rich_text:[]) sono VIETATI → li saltiamo
        - Ogni rich_text.content deve essere ≤ 2000 caratteri
        - heading_2/3 non accettano rich_text vuoto
        """
        LIMIT  = 1900
        blocks = []
        prev_empty = False   # evita paragrafi vuoti consecutivi

        for raw_line in text.splitlines():
            line = raw_line.rstrip()

            if line.startswith("## "):
                content = line[3:].strip()[:LIMIT]
                if content:
                    blocks.append({
                        "object": "block", "type": "heading_2",
                        "heading_2": {"rich_text": [{"type": "text",
                                                     "text": {"content": content}}]},
                    })
                prev_empty = False

            elif line.startswith("### "):
                content = line[4:].strip()[:LIMIT]
                if content:
                    blocks.append({
                        "object": "block", "type": "heading_3",
                        "heading_3": {"rich_text": [{"type": "text",
                                                     "text": {"content": content}}]},
                    })
                prev_empty = False

            elif re.match(r"^[-*]\s+", line):
                content = re.sub(r"^[-*]\s+", "", line).strip()
                for chunk in NotionAPI._split(content, LIMIT):
                    if chunk:
                        blocks.append({
                            "object": "block", "type": "bulleted_list_item",
                            "bulleted_list_item": {"rich_text": [
                                {"type": "text", "text": {"content": chunk}}
                            ]},
                        })
                prev_empty = False

            elif not line:
                # Riga vuota: aggiungi un paragrafo con spazio (non vuoto)
                # ma solo se quello precedente non era già vuoto
                if not prev_empty:
                    blocks.append({
                        "object": "block", "type": "paragraph",
                        "paragraph": {"rich_text": [
                            {"type": "text", "text": {"content": " "}}
                        ]},
                    })
                prev_empty = True

            else:
                for chunk in NotionAPI._split(line, LIMIT):
                    if chunk.strip():
                        blocks.append({
                            "object": "block", "type": "paragraph",
                            "paragraph": {"rich_text": [
                                {"type": "text", "text": {"content": chunk}}
                            ]},
                        })
                prev_empty = False

        return blocks

    @staticmethod
    def _split(text: str, limit: int) -> list[str]:
        """Spezza una stringa in pezzi ≤ limit senza spezzare parole."""
        if len(text) <= limit:
            return [text]
        parts = []
        while text:
            parts.append(text[:limit])
            text = text[limit:]
        return parts

    def archive_page(self, page_id: str):
        self._patch(f"pages/{page_id}", {"archived": True})

    @staticmethod
    def _chunk_text(text: str, limit: int = 1900) -> list[str]:
        """Spezza il testo in pezzi ≤ limit caratteri rispettando i ritorni a capo."""
        chunks, current = [], ""
        for line in text.split("\n"):
            if len(line) > limit:
                if current:
                    chunks.append(current)
                    current = ""
                for i in range(0, len(line), limit):
                    chunks.append(line[i : i + limit])
            elif len(current) + len(line) + 1 > limit:
                chunks.append(current)
                current = line
            else:
                current = (current + "\n" + line) if current else line
        if current:
            chunks.append(current)
        return chunks


# ──────────────────────────────────────────────────────────
# 3. AI ENGINE  (Ollama)
# ──────────────────────────────────────────────────────────
class AIEngine:

    @staticmethod
    def ocr(image_path: Path) -> str:
        """Estrae il testo da un'immagine con GLM-OCR (output tipicamente maiuscolo)."""
        res = ollama.chat(
            model="glm-ocr",
            messages=[{
                "role": "user",
                "content": "Estrai fedelmente tutto il testo presente nell'immagine.",
                "images": [str(image_path)],
            }],
        )
        return res["message"]["content"]

    # Nomi propri italiani comuni nel diritto e nella vita quotidiana
    # sempre maiuscoli anche dopo conversione Python
    _NOMI_PROPRI = {
        "italia", "italiano", "italiana", "italiani", "italiane",
        "costituzione", "parlamento", "senato", "camera", "governo",
        "presidente", "repubblica", "stato", "stati", "corte",
        "unione europea", "europa", "comune", "regione", "provincia",
        "tribunale", "ministero", "consiglio",
    }

    @staticmethod
    def _correct_case_python(raw: str) -> str:
        """
        Conversione rapida TUTTO MAIUSCOLO → italiano normale.
        Istantanea, nessun AI necessario.
        Logica: tutto minuscolo → maiuscola dopo . ! ? e a inizio testo.
        I simboli logici (→ = ≠ ecc.) non sono lettere, restano intatti.
        """
        text = raw.lower()
        # Maiuscola dopo . ! ? seguiti da spazio (gestisce anche virgolette)
        text = re.sub(
            r'([.!?]["\u2019\u2018]?\s+)([a-z\u00e0-\u00f9])',
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )
        # Maiuscola a inizio testo
        if text:
            text = text[0].upper() + text[1:]
        return text

    @staticmethod
    def _clean_formatting(text: str) -> str:
        """
        Normalizza la formattazione del testo corretto:
        - ***Titolo*** o **Titolo** a inizio riga → ## Titolo
        - *Titolo* a inizio riga               → ### Titolo
        - · o • come bullet                    → - 
        - Pulisce asterischi orfani
        """
        lines_out = []
        for line in text.splitlines():
            stripped = line.strip()

            # ***Titolo*** o **Titolo** a inizio riga → heading 2
            m = re.match(r"^\*{2,3}(.+?)\*{2,3}\s*$", stripped)
            if m:
                lines_out.append(f"## {m.group(1).strip()}")
                continue

            # *Titolo* a inizio riga → heading 3
            m = re.match(r"^\*(.+?)\*\s*$", stripped)
            if m:
                lines_out.append(f"### {m.group(1).strip()}")
                continue

            # Bullet point con · o •
            m = re.match(r"^[·•]\s*(.*)", stripped)
            if m:
                lines_out.append(f"- {m.group(1)}")
                continue

            lines_out.append(line)

        return "\n".join(lines_out)

    @staticmethod
    def _rileva_lezione(text: str) -> int | None:
        """
        Cerca nel testo pattern tipo "Lezione 3", "Lez. 3", "Lezione n°3".
        Ritorna il numero intero o None se non trovato.
        """
        m = re.search(
            r"lez(?:ione)?\.?\s*n?[°.]?\s*(\d+)",
            text, re.IGNORECASE
        )
        return int(m.group(1)) if m else None

    @staticmethod
    def correct_case(raw: str) -> str:
        """
        Fase 1 — conversione rapida MAIUSCOLO → italiano normale.
        Python puro, istantanea. La struttura ## viene aggiunta
        da polish_section nella Fase 2.
        """
        return AIEngine._clean_formatting(AIEngine._correct_case_python(raw))


    @staticmethod
    def _parse_json_list(text: str) -> list:
        """Estrae un array JSON dalla risposta del modello, tollerando testo extra."""
        text = text.strip()
        # Prova parsing diretto
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
                # a volte il modello wrappa in {"argomenti": [...]}
                for v in data.values():
                    if isinstance(v, list):
                        return v
        except json.JSONDecodeError:
            pass
        # Cerca il primo array JSON nel testo
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return []

    @staticmethod
    @staticmethod
    def polish_section(titolo: str, corpo: str) -> str:
        """
        Rifinitura finale di una sezione: Qwen corregge grammatica,
        migliora la leggibilità e mantiene la struttura ##/### intatta.
        NON riassume, NON taglia contenuto.
        """
        prompt = (
            f"Stai rifinendo una sezione di appunti universitari"
            f" intitolata: '{titolo}'\n\n"
            "COMPITO:\n"
            "1. Correggi errori grammaticali e refusi.\n"
            "2. Migliora la leggibilita mantenendo il significato.\n"
            "3. Mantieni TUTTA la struttura: ## ### e bullet point (-).\n"
            "4. NON aggiungere introduzioni, conclusioni o commenti.\n"
            "5. NON accorciare: mantieni tutto il contenuto.\n\n"
            "SEZIONE:\n"
            f"{corpo}\n\n"
            "SEZIONE RIFINITA:"
        )
        try:
            res = ollama.chat(
                model=MODEL_SEMANTICA,
                messages=[{"role": "user", "content": prompt}],
            )
            result = res["message"]["content"].strip()
            return result if result else corpo
        except Exception as exc:
            logger.warning(f"polish_section fallito per '{titolo}': {exc}")
            return corpo

    @staticmethod
    def _estrai_titoli(pagine: dict[int, str]) -> str:
        """
        Estrae le prime 4 righe non vuote di ogni pagina per dare
        abbastanza contesto al modello per riconoscere i cambi di argomento.
        """
        righe = []
        for num in sorted(pagine.keys()):
            testo = pagine[num]
            prime = [r.strip() for r in testo.splitlines() if r.strip()][:4]
            corpo = " | ".join(prime) if prime else "(vuota)"
            righe.append(f"Pag {num}: {corpo[:200]}")
        return "\n".join(righe)

    @staticmethod
    def semantic_map(pagine: dict[int, str], total_pages: int) -> list[dict]:
        """
        Analisi semantica veloce:
        1. Estrae solo la prima riga di ogni pagina (titolo/inizio argomento)
        2. Manda l'indice compatto (una riga per pagina) a Qwen — non il testo intero
        3. Chiede una lista semplice "titolo: pag_inizio-pag_fine"
        4. Parsea la risposta con regex Python — nessuna dipendenza da formato JSON
        """
        MODEL = MODEL_SEMANTICA

        # ── Verifica modello ──
        console.print(f"  [dim]Verifica modello {MODEL}…[/dim]")
        try:
            test = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": "Rispondi solo con: ok"}],
            )
            if not test.get("message", {}).get("content", "").strip():
                raise ValueError("Risposta vuota dal modello.")
            console.print(f"  [green]✓  Modello {MODEL} operativo[/green]")
        except Exception as exc:
            console.print(f"  [red]✗  Errore connessione modello: {exc}[/red]")
            console.print(f"  [yellow]Installa il modello con: ollama pull {MODEL}[/yellow]")
            return []

        # ── Costruisce l'indice compatto (solo prima riga per pagina) ──
        indice = AIEngine._estrai_titoli(pagine)
        console.print(f"  [dim]Indice: {len(pagine)} righe inviate a Qwen (niente testo completo)[/dim]")

        p_min = min(pagine.keys())
        p_max = max(pagine.keys())

        prompt = (
            "Sei un assistente che analizza appunti universitari.\n"
            "Ogni riga sotto riporta: numero pagina e le prime righe di quella pagina.\n\n"
            f"{indice}\n\n"
            "COMPITO: individua dove cambia l'argomento trattato e crea un gruppo per ogni argomento.\n\n"
            "REGOLE:\n"
            f"1. Tutte le pagine da {p_min} a {p_max} devono essere incluse.\n"
            "2. VIETATO creare gruppi esattamente di 10 pagine — "
            "devi seguire i cambi di argomento reali nel testo.\n"
            "3. Se un argomento dura 3 pagine, il gruppo è di 3. "
            "Se dura 7, è di 7. Non uniformare.\n"
            "4. Ogni gruppo deve avere un titolo che descrive l'argomento specifico.\n\n"
            "Formato risposta — UNA RIGA per argomento:\n"
            "Titolo argomento: PAG_INIZIO-PAG_FINE\n\n"
            "Esempio di risposta corretta (nota: gruppi di dimensioni diverse):\n"
            "Introduzione e fonti del diritto: 1-4\n"
            "Il Parlamento — composizione: 5-8\n"
            "Il Parlamento — funzioni legislative: 9-13\n"
            "Il Governo: 14-18\n\n"
            "Rispondi SOLO con le righe nel formato indicato, nient'altro."
        )

        for tentativo in range(3):
            try:
                res = ollama.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = res["message"]["content"].strip()
                console.print(f"  [dim]Risposta:\n{raw[:500]}[/dim]")

                risultati = []
                for line in raw.splitlines():
                    m = re.match(r"^(.+?):\s*(\d+)\s*[-–]\s*(\d+)\s*$", line.strip())
                    if m:
                        risultati.append({
                            "titolo":        m.group(1).strip(),
                            "pagina_inizio": int(m.group(2)),
                            "pagina_fine":   int(m.group(3)),
                        })

                if risultati:
                    # ── Controlla e copre le pagine mancanti ──
                    coperte = set()
                    for r in risultati:
                        coperte.update(range(r["pagina_inizio"], r["pagina_fine"] + 1))

                    mancanti = sorted(set(pagine.keys()) - coperte)
                    if mancanti:
                        console.print(
                            f"  [yellow]⚠  Pagine non coperte: {mancanti} "
                            f"→ aggiunte come argomento separato[/yellow]"
                        )
                        # Raggruppa le pagine mancanti in blocchi contigui
                        blocchi_mancanti = []
                        start = mancanti[0]
                        prev  = mancanti[0]
                        for p in mancanti[1:]:
                            if p != prev + 1:
                                blocchi_mancanti.append((start, prev))
                                start = p
                            prev = p
                        blocchi_mancanti.append((start, prev))

                        for b_start, b_end in blocchi_mancanti:
                            # Usa il titolo della prima pagina del blocco
                            prime = [r.strip() for r in pagine[b_start].splitlines()
                                     if r.strip()][:1]
                            titolo_gap = prime[0][:60] if prime else f"Pagine {b_start}-{b_end}"
                            risultati.append({
                                "titolo":        titolo_gap,
                                "pagina_inizio": b_start,
                                "pagina_fine":   b_end,
                            })
                        risultati.sort(key=lambda x: x["pagina_inizio"])

                    console.print(f"  [green]✓  {len(risultati)} argomenti (copertura {p_min}-{p_max})[/green]")
                    return risultati

                console.print(
                    f"  [yellow]⚠  Tentativo {tentativo+1}: nessun argomento parsato.\n"
                    f"  Risposta: {raw[:300]}[/yellow]"
                )
            except Exception as exc:
                console.print(f"  [red]✗  Tentativo {tentativo+1} fallito: {exc}[/red]")
            time.sleep(2)

        return []

# ──────────────────────────────────────────────────────────
# 4. PIPELINE PRINCIPALE
# ──────────────────────────────────────────────────────────
class DocumentPipeline:
    def __init__(self):
        self.api = NotionAPI(NOTION_TOKEN)
        self.ai  = AIEngine()

    # ── Utilità ─────────────────────────────────
    @staticmethod
    def _norm(text: str) -> str:
        """
        Normalizza una stringa per confronti case-insensitive e accent-insensitive.
        Converte tutti i tipi di apostrofo/accento tipografico in apostrofo semplice
        prima di normalizzare: i titoli Notion usano spesso ' (curvo) invece di ' (dritto),
        il che causerebbe mancate corrispondenze senza questa sostituzione.
        """
        # Sostituisce varianti tipografiche di apostrofo e accento grave
        for ch in ("\u2019", "\u2018", "\u02BC", "\u0060", "\u00B4"):
            text = text.replace(ch, "'")
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ASCII", "ignore")
            .decode("utf-8")
            .lower()
            .strip()
        )

    @staticmethod
    def _needs_rewrite(text: str) -> bool:
        """
        Ritorna True se la pagina ha bisogno di essere riscritta da Qwen.
        Casi:
        - Testo tutto MAIUSCOLO (OCR grezzo)
        - Testo tutto minuscolo senza maiuscole a inizio frase (Python-corretto male)
        - Nessun titolo ## presente (nessuna struttura)
        """
        letters = [c for c in text if c.isalpha()]
        if len(letters) < 20:
            return False
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        # Tutto maiuscolo
        if upper_ratio > 0.60:
            return True
        # Tutto minuscolo (Python ha solo lowercasato senza strutturare)
        if upper_ratio < 0.02:
            return True
        # Nessun titolo di sezione
        if "## " not in text:
            return True
        return False

    @staticmethod
    def _is_uppercase(text: str, threshold: float = 0.60) -> bool:
        """Compatibilità: usa _needs_rewrite."""
        return DocumentPipeline._needs_rewrite(text)

    # ── Navigazione Notion ───────────────────────
    def trova_database(self, subject: str) -> str:
        """Trova il database 'Note/Appunti' del corso specificato."""
        console.print(f"\n[cyan]📡 Ricerca corso [bold]'{subject}'[/bold] su Notion…[/cyan]")
        courses = self.api.query_db(NOTION_ROOT_PAGE_ID)

        match = next(
            (c for c in courses
            if self._norm(subject) in self._norm(
                c.get("properties", {})
                .get("Name", {})
                .get("title", [{}])[0]
                .get("plain_text", "")
            )),
            None,
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
            blocks = self.api.fetch_blocks(parent_id)

            # Match diretto a questo livello: child_database pertinente
            for b in blocks:
                if b.get("type") == "child_database" and any(
                    k in self._norm(b.get("child_database", {}).get("title", ""))
                    for k in keywords
                ):
                    return b

            # Nessun match diretto: scendi nei contenitori strutturali
            # (child_page, ma anche column_list/column che Notion usa per i layout a bottoni)
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

    # ── FASE 1: Controllo, Correzione e Upload batch ──
    def fase1_acquisizione(self, doc_path: Path, database_id: str):
        """
        Per ogni batch di 10 pagine:
          A) Pagina Notion esiste ed è maiuscola → leggi testo, correggi con Qwen,
             archivia la vecchia, carica quella corretta.
          B) Pagina Notion esiste ed è già corretta → salto.
          C) Pagina Notion non esiste → OCR dal PDF → correggi → carica.
        Il testo corretto viene salvato in cache locale per la Fase 2.
        """
        pdf  = fitz.open(doc_path)
        tot  = len(pdf)
        stem = doc_path.stem

        console.print(Panel(
            f"[bold]FASE 1 — Controllo e Correzione[/bold]\n\n"
            f"  Documento : [cyan]{doc_path.name}[/cyan]\n"
            f"  Pagine    : [yellow]{tot}[/yellow]\n"
            f"  Gruppi    : [yellow]{BATCH_SIZE}[/yellow] pag / batch",
            border_style="blue",
        ))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            ) as prog:
                task_glob = prog.add_task("[cyan]Batch totali", total=(tot // BATCH_SIZE + (1 if tot % BATCH_SIZE else 0)))

                for batch_start in range(1, tot + 1, BATCH_SIZE):
                    batch_end   = min(batch_start + BATCH_SIZE - 1, tot)
                    batch_num   = (batch_start - 1) // BATCH_SIZE + 1
                    batch_title = f"{stem} - Parte {batch_num} (Pag {batch_start}-{batch_end})"

                    prog.print(Rule(
                        f"[bold blue]Batch {batch_num}  (pag. {batch_start}–{batch_end})[/bold blue]",
                        style="blue"
                    ))

                    # ── Controlla se la pagina esiste su Notion ──
                    notion_page = self.api.get_page_by_title(database_id, batch_title)

                    if notion_page:
                        testo_notion = self.api.fetch_page_text(notion_page["id"])

                        if not self._is_uppercase(testo_notion):
                            # ── CASO B: già corretta ──
                            prog.print(f"  [green]⏭  Già corretta su Notion → salto[/green]")
                            # Salva in cache locale per la Fase 2 (se non c'è già)
                            self._salva_pagine_in_cache(stem, batch_start, batch_end, testo_notion)
                            prog.advance(task_glob, 1)
                            continue

                        # ── CASO A: esiste ma è maiuscola → correggi ──
                        prog.print(
                            f"  [yellow]⚠  Testo maiuscolo rilevato "
                            f"→ correzione con Qwen…[/yellow]"
                        )
                        testo_corretto = self.ai.correct_case(testo_notion)
                        lezione = self.ai._rileva_lezione(testo_corretto)
                        if lezione:
                            prog.print(f"  [dim]Lezione rilevata: {lezione}[/dim]")

                        # Archivia la versione maiuscola
                        try:
                            self.api.archive_page(notion_page["id"])
                        except Exception:
                            pass

                        # Carica la versione corretta
                        try:
                            self.api.create_page(database_id, batch_title,
                                                 testo_corretto, lezione=lezione)
                            prog.print(f"  [green]✓  Versione corretta caricata su Notion[/green]")
                        except Exception as exc:
                            prog.print(f"  [red]✗  Errore upload: {exc}[/red]")
                            logger.error(f"Errore upload {batch_title}: {exc}")

                        # Salva in cache locale per la Fase 2
                        self._salva_pagine_in_cache(stem, batch_start, batch_end, testo_corretto)
                        logger.info(f"Batch corretto e ricaricato: {batch_title}")

                    else:
                        # ── CASO C: pagina assente → OCR dal PDF ──
                        prog.print(f"  [blue]📄  Pagina assente su Notion → OCR dal PDF[/blue]")
                        testo_batch = ""

                        task_ocr = prog.add_task(
                            f"[white]  OCR batch {batch_num}",
                            total=batch_end - batch_start + 1,
                        )
                        for num in range(batch_start, batch_end + 1):
                            c_grezzo = DIR_CACHE_GREZZO / f"{stem}_p{num}.txt"
                            c_pulito = DIR_CACHE_PULITO / f"{stem}_p{num}.txt"

                            if c_pulito.exists() and not self._is_uppercase(
                                c_pulito.read_text(encoding="utf-8")
                            ):
                                t_pulito = c_pulito.read_text(encoding="utf-8")
                                prog.print(f"  [dim]P.{num:02d}[/dim]  [green]Cache ✓[/green]")
                            else:
                                if not c_grezzo.exists():
                                    prog.print(f"  [dim]P.{num:02d}[/dim]  [blue]OCR…[/blue]")
                                    img_path = tmp_path / f"p{num}.png"
                                    pdf.load_page(num - 1).get_pixmap(dpi=200).save(img_path)
                                    t_grezzo = self.ai.ocr(img_path)
                                    c_grezzo.write_text(t_grezzo, encoding="utf-8")
                                else:
                                    t_grezzo = c_grezzo.read_text(encoding="utf-8")

                                prog.print(f"  [dim]P.{num:02d}[/dim]  [magenta]Correzione…[/magenta]")
                                t_pulito = self.ai.correct_case(t_grezzo)
                                c_pulito.write_text(t_pulito, encoding="utf-8")

                            testo_batch += f"\n\n## --- Pagina {num} ---\n\n{t_pulito}"
                            prog.advance(task_ocr, 1)

                        lezione = self.ai._rileva_lezione(testo_batch)
                        try:
                            self.api.create_page(database_id, batch_title,
                                                 testo_batch, lezione=lezione)
                            prog.print(f"  [green]✓  Caricato:[/green] '{batch_title}'"
                                       + (f" (Lezione {lezione})" if lezione else ""))
                        except Exception as exc:
                            prog.print(f"  [red]✗  Errore upload: {exc}[/red]")

                    prog.advance(task_glob, 1)

        console.print(Panel(
            "[bold green]✅ FASE 1 completata[/bold green]\n"
            "Tutte le pagine sono state elaborate e caricate su Notion.",
            border_style="green",
        ))

    @staticmethod
    def _salva_pagine_in_cache(stem: str, start: int, end: int, testo_batch: str):
        """Estrae e salva ogni pagina del batch nella cache locale per la Fase 2."""
        for num in range(start, end + 1):
            c_pulito = DIR_CACHE_PULITO / f"{stem}_p{num}.txt"
            if not c_pulito.exists():
                pat = rf"--- Pagina {num} ---(.*?)(?=---[ \t]*Pagina[ \t]*\d|$)"
                m   = re.search(pat, testo_batch, re.DOTALL)
                testo_pagina = m.group(1).strip() if m else testo_batch
                c_pulito.write_text(testo_pagina, encoding="utf-8")

    # ── FASE 2: Segmentazione per titoli di sezione ─────────
    # ── FASE 2: Segmentazione per sezioni ─────────
    # ── FASE 2: Segmentazione Semantica ─────────
    # ── FASE 2: Segmentazione Semantica ─────────
    def fase2_segmentazione(self, doc_path: Path, database_id: str):
        stem      = doc_path.stem
        stato_path = Path(f"stato_fase2_{stem}.json")

        # ── Carica stato precedente (se esiste) ──
        stato = {"mappa": [], "caricati": [], "archiviate": False}
        if stato_path.exists():
            try:
                stato = json.loads(stato_path.read_text(encoding="utf-8"))
                console.print(f"[cyan]♻  Stato precedente trovato: "
                               f"{len(stato['caricati'])} gruppi già caricati[/cyan]")
            except Exception:
                pass

        def salva_stato():
            stato_path.write_text(json.dumps(stato, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

        console.print(Panel(
            "[bold]FASE 2 — Segmentazione Semantica[/bold]\n\n"
            "  1. Carica pagine dalla cache\n"
            "  2. Qwen raggruppa per argomento (solo titoli)\n"
            "  3. Carica su Notion\n"
            "  4. Archivia vecchie pagine (solo se tutto OK)",
            border_style="magenta",
        ))

        # ── 1. Carica pagine dalla cache ──
        console.print("\n[cyan]📂 Caricamento cache…[/cyan]")
        _PAT = re.compile(r"_p(\d+)\.txt$")
        cache_files = sorted(
            (f for f in DIR_CACHE_PULITO.glob(f"{stem}_p*.txt") if _PAT.search(f.name)),
            key=lambda f: int(_PAT.search(f.name).group(1)),
        )
        if not cache_files:
            console.print("[red]✗  Cache non trovata. Esegui prima la Fase 1.[/red]")
            return

        pagine: dict[int, str] = {}
        for f in cache_files:
            num = int(_PAT.search(f.name).group(1))
            pagine[num] = f.read_text(encoding="utf-8").strip()
        console.print(f"  [green]✓  {len(pagine)} pagine caricate[/green]")

        # ── 2. Mappa semantica (riusa quella salvata se già fatta) ──
        if stato["mappa"]:
            mappa = stato["mappa"]
            console.print(f"  [dim]Mappa semantica recuperata dallo stato ({len(mappa)} gruppi)[/dim]")
        else:
            console.print("\n[magenta]🧠 Analisi semantica (solo titoli → Qwen)…[/magenta]")

            indice_righe = []
            for num in sorted(pagine.keys()):
                prime = [r.strip().lstrip("#").strip()
                         for r in pagine[num].splitlines() if r.strip()][:3]
                indice_righe.append(f"Pag {num}: {' | '.join(prime)[:180]}")
            indice = "\n".join(indice_righe)

            p_min, p_max = min(pagine.keys()), max(pagine.keys())

            prompt = (
                "Sei un assistente che organizza appunti universitari.\n"
                "Qui sotto c'è un indice: numero pagina e le prime righe di contenuto.\n\n"
                f"{indice}\n\n"
                "COMPITO: raggruppa le pagine in macro-argomenti tematici.\n"
                "Argomenti come 'Camera', 'Senato', 'iter legislativo' fanno parte\n"
                "dello stesso macro-argomento 'Il Parlamento'.\n\n"
                "REGOLE:\n"
                f"1. Ogni pagina da {p_min} a {p_max} deve essere in esattamente un gruppo.\n"
                "2. Segui i cambi di tema reali, non i numeri di pagina.\n"
                "3. Gruppi di dimensioni variabili (3-15 pagine).\n"
                "4. Titoli dei gruppi: specifici e descrittivi.\n\n"
                "FORMATO RISPOSTA — una riga per gruppo:\n"
                "Titolo gruppo: PAG_INIZIO-PAG_FINE\n\n"
                "Esempio:\n"
                "Il Parlamento — composizione e funzioni: 1-12\n"
                "Il Governo: 13-19\n"
                "Le fonti del diritto: 20-28\n\n"
                "Rispondi SOLO con le righe nel formato indicato."
            )

            mappa = []
            for tentativo in range(3):
                try:
                    res = ollama.chat(model=MODEL_SEMANTICA,
                                     messages=[{"role": "user", "content": prompt}])
                    raw = res["message"]["content"].strip()
                    console.print(f"  [dim]Risposta Qwen:\n{raw}[/dim]")
                    for line in raw.splitlines():
                        m = re.match(r"^(.+?):\s*(\d+)\s*[-–]\s*(\d+)\s*$", line.strip())
                        if m:
                            mappa.append({"titolo": m.group(1).strip(),
                                          "pagina_inizio": int(m.group(2)),
                                          "pagina_fine":   int(m.group(3))})
                    if mappa:
                        break
                    console.print(f"  [yellow]⚠  Tentativo {tentativo+1}: nessun gruppo parsato[/yellow]")
                except Exception as exc:
                    console.print(f"  [red]✗  Tentativo {tentativo+1}: {exc}[/red]")
                time.sleep(2)

            if not mappa:
                console.print("[red]✗  Impossibile generare la mappa.[/red]")
                return

            # Copertura
            p_min, p_max = min(pagine.keys()), max(pagine.keys())
            coperte = set()
            for g in mappa:
                coperte.update(range(g["pagina_inizio"], g["pagina_fine"] + 1))
            mancanti = sorted(set(pagine.keys()) - coperte)
            if mancanti:
                console.print(f"  [yellow]⚠  Pagine non coperte {mancanti} → aggiunte all'ultimo gruppo[/yellow]")
                mappa[-1]["pagina_fine"] = max(mappa[-1]["pagina_fine"], max(mancanti))
            mappa.sort(key=lambda x: x["pagina_inizio"])

            stato["mappa"] = mappa
            salva_stato()

        # Mostra mappa
        tbl = Table(title="📊 Mappa Semantica", box=box.ROUNDED,
                    header_style="bold magenta", show_lines=True)
        tbl.add_column("N.", justify="right", width=4)
        tbl.add_column("Argomento", min_width=40)
        tbl.add_column("Inizio", justify="center", width=7)
        tbl.add_column("Fine",   justify="center", width=6)
        tbl.add_column("Pag.",   justify="center", width=5)
        for i, g in enumerate(mappa, 1):
            tbl.add_row(str(i), g["titolo"],
                        str(g["pagina_inizio"]), str(g["pagina_fine"]),
                        str(g["pagina_fine"] - g["pagina_inizio"] + 1))
        console.print(tbl)

        # ── 3. Carica su Notion (salta già caricati) ──
        console.print("\n[cyan]☁  Caricamento su Notion…[/cyan]")
        successi = len(stato["caricati"])
        errori   = 0
        p_min, p_max = min(pagine.keys()), max(pagine.keys())

        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      BarColumn(), MofNCompleteColumn(),
                      console=console, transient=False) as prog:
            task = prog.add_task("[cyan]Argomenti", total=len(mappa))
            prog.advance(task, successi)   # mostra i già completati

            for g in mappa:
                titolo = (g["titolo"]
                          .replace("/", "-").replace("\\", "-")
                          .replace(":", " —").strip())

                if titolo in stato["caricati"]:
                    prog.print(f"  [dim]⏭  Già caricato: '{titolo}'[/dim]")
                    prog.advance(task, 1)
                    continue

                p_in  = max(p_min, g["pagina_inizio"])
                p_fin = min(p_max, g["pagina_fine"])
                testo = "\n\n".join(pagine[p] for p in range(p_in, p_fin+1) if p in pagine)

                if not testo.strip():
                    prog.advance(task, 1)
                    continue

                try:
                    self.api.create_page(database_id, titolo, testo)
                    prog.print(f"  [green]✓[/green]  '{titolo}'  (pag. {p_in}–{p_fin})")
                    successi += 1
                    stato["caricati"].append(titolo)
                    salva_stato()          # ← checkpoint dopo ogni pagina caricata
                except Exception as exc:
                    prog.print(f"  [red]✗[/red]  '{titolo}' → {str(exc)[:120]}")
                    errori += 1

                prog.advance(task, 1)

        # ── 4. Archivia solo se tutto OK ──
        if errori == 0 and successi > 0 and not stato["archiviate"]:
            console.print("\n[cyan]🧹 Archiviazione pagine precedenti…[/cyan]")
            bozze = self.api.get_pages_starting_with(database_id, f"{stem} - Parte")
            for b in bozze:
                try:
                    self.api.archive_page(b["id"])
                except Exception:
                    pass
            stato["archiviate"] = True
            salva_stato()
            console.print("  [green]✓  Bozze archiviate[/green]")
        elif errori > 0:
            console.print(f"\n[yellow]⚠  {errori} errori — bozze NON archiviate per sicurezza.[/yellow]")

        # Pulizia file di stato se completato
        if errori == 0 and stato["archiviate"]:
            stato_path.unlink(missing_ok=True)
            console.print("  [dim]File di stato rimosso (processo completato)[/dim]")

        console.print(Panel(
            f"[bold green]✅ FASE 2 completata[/bold green]\n\n"
            f"  Caricati : [green]{successi}[/green] argomenti\n"
            f"  Errori   : [red]{errori}[/red]",
            border_style="green",
        ))

    @staticmethod
    @staticmethod
    def _estrai_sezioni(testo: str) -> list[tuple[str, str]]:
        """Estrae sezioni da titoli ## — non usata nella logica principale."""
        sezioni, titolo_corrente, righe_corpo = [], None, []
        PAT_MARKER = re.compile(r"^-+\s*pagina\s*\d+\s*-+$", re.IGNORECASE)
        for riga in testo.splitlines():
            s = riga.strip()
            if not s:
                continue
            if s.startswith("## "):
                cand = s[3:].strip()
                if PAT_MARKER.match(cand) or len(cand) < 3:
                    continue
                if titolo_corrente:
                    corpo = "\n".join(righe_corpo).strip()
                    if len(corpo) > 50:
                        sezioni.append((titolo_corrente, corpo))
                titolo_corrente, righe_corpo = cand, []
            elif titolo_corrente is not None:
                righe_corpo.append(riga)
        if titolo_corrente:
            corpo = "\n".join(righe_corpo).strip()
            if len(corpo) > 50:
                sezioni.append((titolo_corrente, corpo))
        return sezioni

# ──────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ──────────────────────────────────────────────────────────
def main():
    console.print(Panel(
        "[bold cyan]🔬 Pipeline OCR → Notion[/bold cyan]\n"
        "Digitalizzazione appunti con segmentazione semantica AI",
        subtitle="v2.0",
        border_style="cyan",
    ))

    # ── Input utente ──
    raw_path = console.input(
        "\n[bold] Percorso file PDF (o cartella con PDF): [/bold]"
    ).strip().strip("\"'")
    target = Path(raw_path).expanduser().resolve()

    if not target.exists():
        console.print(f"[red]✗  Percorso non trovato: {target}[/red]")
        return

    subject = console.input("[bold] Materia del corso su Notion: [/bold]").strip()

    console.print("\n[bold]Seleziona operazione:[/bold]")
    console.print("  [cyan]1[/cyan]  Fase 1 + Fase 2  (pipeline completa)")
    console.print("  [cyan]2[/cyan]  Solo Fase 1  (OCR → correzione → upload batch)")
    console.print("  [cyan]3[/cyan]  Solo Fase 2  (segmentazione semantica → argomenti)")
    scelta = console.input("[bold] Scelta [1/2/3]: [/bold]").strip()

    console.rule()

    # ── Connessione Notion ──
    pipeline = DocumentPipeline()
    try:
        db_id = pipeline.trova_database(subject)
    except Exception as exc:
        console.print(f"[red]✗  Errore navigazione Notion: {exc}[/red]")
        return

    # ── Raccolta PDF ──
    docs = (
        [target] if target.is_file()
        else sorted(f for f in target.iterdir() if f.suffix.lower() == ".pdf")
    )
    if not docs:
        console.print("[red]✗  Nessun file PDF trovato.[/red]")
        return

    console.print(f"\n[green]📄  {len(docs)} documento/i trovato/i[/green]")

    # ── Elaborazione ──
    for doc in docs:
        console.print(Rule(f"[bold]{doc.name}[/bold]"))
        try:
            if scelta in ("1", "2"):
                pipeline.fase1_acquisizione(doc, db_id)
            if scelta in ("1", "3"):
                pipeline.fase2_segmentazione(doc, db_id)
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠  Interruzione manuale.[/yellow]")
            break
        except Exception as exc:
            console.print(f"[red]✗  Errore su '{doc.name}': {exc}[/red]")
            logger.exception(f"Errore {doc.name}")

    console.print(Panel(
        "[bold green]🎉 Pipeline terminata![/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
