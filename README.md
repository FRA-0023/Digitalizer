# OCR Pipeline - Digitalizer

Applicazione Python per l'estrazione e digitalizzazione di testo da documenti PDF, con integrazione con Notion e gestione intelligente della cache.

## 📋 Descrizione

**Digitalizer** è un bot OCR che automatizza il processo di estrazione di testo da file PDF. L'applicazione:

- 🎯 Estrae testo da documenti PDF utilizzando OCR avanzato
- 💾 Mantiene cache locali per ottimizzare le risorse computazionali
- 🔗 Integra con Notion per l'organizzazione e la gestione dei documenti
- 🧹 Normalizza e pulisce il testo estratto
- 📁 Organizza i risultati in cartelle di cache (grezzo e revisionato)

## 🚀 Caratteristiche

- **OCR Intelligente**: Utilizza Ollama per l'estrazione di testo ad alta precisione
- **Cache Locale**: Evita re-elaborazione di documenti già processati
  - `cache_ocr_grezzo/`: Testo grezzo estratto
  - `cache_ocr_revisionato/`: Testo normalizzato e pulito
- **Integrazione Notion**: Sincronizza direttamente con database Notion
- **Normalizzazione Testo**: Rimuove caratteri speciali, normalizza Unicode e lowercasing
- **Gestione Gerarchica**: Naviga tra pagine, database e blocchi di contenuto in Notion

## 📦 Prerequisiti

- Python 3.8+
- Ollama installato e in esecuzione (per l'OCR)
- Account Notion con API Token
- pip (Python package manager)

## 🔧 Installazione

1. **Clona o scarica il progetto**
```bash
cd Digitalizer
```

2. **Crea un ambiente virtuale (consigliato)**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

3. **Installa le dipendenze**
```bash
pip install -r requirements.txt
```

## ⚙️ Configurazione

1. **Crea un file `.env` nella root del progetto**
```bash
touch .env
```

2. **Popola le variabili d'ambiente**
```
NOTION_TOKEN=your_notion_api_token_here
NOTION_ROOT_PAGE_ID=your_notion_page_id_here
```

### Come ottenere le credenziali Notion

- **NOTION_TOKEN**: 
  1. Vai a [https://www.notion.so/my-integrations](https://www.notion.so/my-integrations)
  2. Crea una nuova integrazione
  3. Copia il "Internal Integration Token"

- **NOTION_ROOT_PAGE_ID**:
  1. Apri la pagina Notion desiderata
  2. Copia l'ID dalla URL (es: `https://notion.so/workspace/PAGE_ID?v=xyz`)

## 📊 Struttura del Progetto

```
Digitalizer/
├── ocr_pipeline.py           # Script principale
├── .env                       # Variabili d'ambiente (da creare)
├── .gitignore                # File Git da ignorare
├── cache_ocr_grezzo/         # Cache testo grezzo OCR
├── cache_ocr_revisionato/    # Cache testo normalizzato
└── README.md                 # Questo file
```

## 🎬 Utilizzo

### Eseguire il pipeline

```bash
python ocr_pipeline.py
```

Il programma ti guiderà attraverso:
1. Selezione del corso/progetto da Notion
2. Scelta del documento da processare
3. Estrazione e normalizzazione del testo
4. Sincronizzazione con Notion

## 📚 Dipendenze Principali

- **ollama**: Client per il modello OCR Ollama
- **fitz (PyMuPDF)**: Elaborazione file PDF
- **requests**: Chiamate HTTP per Notion API
- **python-dotenv**: Gestione variabili d'ambiente
- **pathlib**: Gestione percorsi file cross-platform

## 🔍 Funzioni Principali

### `normalizza_testo(testo: str) -> str`
Normalizza il testo estratto:
- Converte caratteri speciali (es: `'` → `'`)
- Normalizza Unicode (NFKD)
- Converte in ASCII
- Converte in minuscolo

### `fetch_database_entries(database_id: str) -> list[dict]`
Recupera tutte le entry da un database Notion con paginazione.

### `fetch_children_as_targets(parent_id: str) -> list[dict]`
Naviga ricorsivamente la struttura Notion fino a trovare database e pagine.

### `navigate_to_database(subject: str) -> tuple[str, str, str]`
Interfaccia interattiva per navigare e selezionare database su Notion.

## 💡 Tips & Tricks

- **Cache Hit**: Se un documento è già stato processato, il sistema lo riutilizzerà
- **Ottimizzazione**: Ollama deve essere in esecuzione prima di avviare il pipeline
- **Debug**: Controlla i file di cache per verificare l'output intermedio

## ⚠️ Troubleshooting

### Errore: "NOTION_TOKEN e NOTION_ROOT_PAGE_ID non configurati"
- Verifica di aver creato correttamente il file `.env`
- Assicurati che le variabili siano scritte esattamente come sopra

### Errore: "Nessun modello OCR disponibile"
- Verifica che Ollama sia in esecuzione: `ollama serve`
- Controlla di aver installato un modello OCR: `ollama pull <model-name>`

### PDF non leggibile
- Verifica che il file PDF non sia corrotto
- Prova con un altro file PDF per escludere problemi con il documento

## 📝 Note

- Il progetto è compatibile con Windows, macOS e Linux
- La cache locale riduce significativamente i tempi di elaborazione
- I file `.env` sono ignorati da Git per motivi di sicurezza (check `.gitignore`)

## 📞 Supporto

Per problemi o suggerimenti, verifica:
1. La connessione a Notion e Ollama
2. I permessi del database Notion
3. La sintassi del file `.env`

---

**Versione**: 1.0  
**Ultima modifica**: 2026
