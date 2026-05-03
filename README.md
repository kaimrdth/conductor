# Conductor | A Local Agentic Harness for Qwen3

![Conductor Screenshot](assets/screencap.png)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-Supported-orange.svg)](https://ollama.com/)
[![Model: Qwen3.5:9b](https://img.shields.io/badge/Model-Qwen3.5:9b-green.svg)](https://qwenlm.github.io/)

Conductor is a lightweight, command-line harness designed for **local LLM inference**. It transforms local models (specifically optimized for Qwen3 via Ollama) into capable, autonomous agents utilizing a ReAct (Reason and Act) loop. This was built primarily as an experiment to test the upper bound of running local inference with a harness on a decent-ish laptop.

**Hardware Context:** This harness was explicitly built, constrained, and tested on a **2021 MacBook Pro (M1 chip, 16GB RAM)**. It demonstrates that sophisticated, agentic workflows—including web scraping, RAG, and file manipulation—can run securely and efficiently entirely on consumer hardware, without relying on cloud APIs.

## Key Features

* **ReAct Architecture:** Alternates between thought, tool execution, and observation to iteratively solve problems (capped at 8 steps per turn to prevent infinite loops).
* **Strict Sandboxing:** All file operations are strictly confined to a local `./workspace` directory to prevent unauthorized system modifications.
* **Built-in Tools:**
  * **File System:** `read_file`, `write_file`, and `list_files`.
  * **Web Fetching:** `fetch_url` extracts relevant text from web pages, utilizing local RAG (Retrieval-Augmented Generation) chunking.
  * **PDF Extraction:** Built-in support for reading PDFs (requires `pdfplumber` or `pypdf`).
  * **Shell Commands:** `run_command` executes a curated allowlist of shell commands (`obsidian`, `git`, `ls`, `pwd`, `echo`, `ps`, `df`, `date`, `python3`) from the workspace directory. Commands outside the allowlist are blocked outright; `python3 -c` is also blocked to prevent arbitrary code execution.
* **Multi-Mode Operation:**
  * **Default Mode:** Conversational interaction where Conductor can immediately act and use tools.
  * **Plan Mode (`/plan`):** A read-only phase that follows a four-step process — **Explore** (read workspace), **Analyze** (identify changes and risks), **Plan** (write structured Markdown plan to `workspace/plan.md`), **Signal** (emit `<PLAN_READY/>` when complete). No writes are permitted during this phase.
  * **Execute Mode (`/approve`):** Reviews and executes the previously generated plan step-by-step.
* **Persistent Memory:** Context is continuously updated and saved across sessions via a `state.md` file.
* **Obsidian Vault Integration:** If an Obsidian vault path is stored in `state.md`, file operations and `run_command obsidian` calls are also permitted against vault files, in addition to the standard `./workspace` sandbox.
* **First-Run Onboarding:** On the first launch, Conductor prompts for your name, Obsidian vault path, and any additional context to pre-populate `state.md`. All fields are optional.
* **ESC Interrupt:** Press `ESC` at any time while the agent is running to cancel the current turn mid-stream. The turn is discarded and the prompt returns immediately.
* **Voice Input (Push-to-Talk):** Hold right ⌥ (Option) to record, release to transcribe and send. Text is injected directly at the prompt. Falls back to `/mic` for manual record-then-Enter input. Requires `pynput`, `sounddevice`, and `mlx-whisper`; PTT needs a one-time macOS Accessibility permission grant.
* **Write Approval with Diff Preview:** Before any file is written, Conductor shows a line-level diff (green highlights for additions, red for removals) and prompts `y / a (approve all) / n`. No file is touched without explicit confirmation.
* **Visual Progress Indicators:** A grey dot appears on tool start, replaced by a green dot on completion. Chained tool calls are displayed hierarchically and collapsed after 3 with a "+N more tool uses" summary.
* **Qwen3 Optimized:** Intelligently handles Qwen3's thinking tokens, stripping them from the assistant history to save context space, while utilizing `/no_think` prompts for tool-execution turns.

## Prerequisites

* Python 3.8+
* [Ollama](https://ollama.com/) running locally.
* The Qwen3 model installed in Ollama. The default configuration uses `qwen3.5:9b`:
  ```bash
  ollama run qwen3.5:9b
  ```

## Installation

1. Clone this repository or place `conductor.py` in your desired directory.
2. Install the core Python dependencies:
   ```bash
   pip install httpx beautifulsoup4 pydantic rich ollama
   ```
3. *(Optional)* Install extended dependencies for PDF parsing and advanced embedding retrieval:
   ```bash
   pip install pdfplumber pypdf sentence-transformers
   ```
4. *(Optional)* Install voice input dependencies (push-to-talk + `/mic`):
   ```bash
   pip install mlx-whisper sounddevice pynput
   ```
   > **macOS:** PTT requires granting Accessibility permission to your terminal app — System Settings → Privacy & Security → Accessibility.

## Usage

Start the interactive terminal application:

```bash
python conductor.py
```

### Interactive Commands

Inside the Conductor interface, you can use the following commands to manage the agent:

* `/plan` — Toggle **Plan Mode** (read-only). Use this to let Conductor map out a solution safely.
* `/approve` — Execute the plan currently saved in `workspace/plan.md`.
* `/state` — View the current persistent memory state.
* `/compact [focus]` — Instructs the LLM to compress the current conversation history into a dense summary, freeing up the context window.
* `/mic` — Fallback voice input: starts recording immediately, press Enter to stop and transcribe.
* `/clear` — Clear the active conversation history (does not delete the persistent `state.md`).
* `/help` — Display the list of commands.
* `/exit` or `/quit` — Exit the application and display session statistics (time elapsed, tokens used, tool calls made).

## Research & Security Basis

Conductor incorporates several modern LLM development patterns:

* **ReAct Pattern:** Based on Yao et al. (2022) for alternating thought/action.
* **Prompt Injection Defense:** Follows OWASP LLM Top 10 guidelines by explicitly wrapping web data as untrusted content.
* **Local RAG Chunking:** Optimized 400-word chunks with 50-word overlaps using `multi-qa-mpnet-base-dot-v1` embeddings (preferred over MiniLM for retrieval tasks).