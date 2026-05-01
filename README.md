Conductor — Local Agentic Harness for Qwen3

Conductor is a lightweight, command-line harness designed for local LLM inference. It transforms local models (specifically optimized for Qwen3 via Ollama) into capable, autonomous agents utilizing a ReAct (Reason and Act) loop.

Hardware Context: This harness was explicitly built, constrained, and tested on a 2021 MacBook Pro (M1 chip, 16GB RAM). It demonstrates that sophisticated, agentic workflows—including web scraping, RAG, and file manipulation—can run securely and efficiently entirely on consumer hardware, without relying on cloud APIs.

Key Features

ReAct Architecture: Alternates between thought, tool execution, and observation to iteratively solve problems (capped at 8 steps per turn to prevent infinite loops).

Strict Sandboxing: All file operations are strictly confined to a local ./workspace directory to prevent unauthorized system modifications.

Built-in Tools:

File System: read_file, write_file, and list_files.

Web Fetching: fetch_url extracts relevant text from web pages, utilizing local RAG (Retrieval-Augmented Generation) chunking.

PDF Extraction: Built-in support for reading PDFs (requires pdfplumber or pypdf).

Multi-Mode Operation:

Default Mode: Conversational interaction where Conductor can immediately act and use tools.

Plan Mode (/plan): A read-only phase where the agent explores the workspace, analyzes the task, and outputs a structured Markdown plan (workspace/plan.md) without making changes.

Execute Mode (/approve): Reviews and executes the previously generated plan step-by-step.

Persistent Memory: Context is continuously updated and saved across sessions via a state.md file.

Qwen3 Optimized: Intelligently handles Qwen3's thinking tokens, stripping them from the assistant history to save context space, while utilizing /no_think prompts for tool-execution turns.

Prerequisites

Python 3.8+

Ollama running locally.

The Qwen3 model installed in Ollama. The default configuration uses qwen3.5:9b:

ollama run qwen3.5:9b


Installation

Clone this repository or place conductor.py in your desired directory.

Install the core Python dependencies:

pip install httpx beautifulsoup4 pydantic rich ollama


(Optional) Install extended dependencies for PDF parsing and advanced embedding retrieval:

pip install pdfplumber pypdf sentence-transformers


Usage

Start the interactive terminal application:

python conductor.py


Interactive Commands

Inside the Conductor interface, you can use the following commands to manage the agent:

/plan — Toggle Plan Mode (read-only). Use this to let Conductor map out a solution safely.

/approve — Execute the plan currently saved in workspace/plan.md.

/state — View the current persistent memory state.

/compact [focus] — Instructs the LLM to compress the current conversation history into a dense summary, freeing up the context window.

/clear — Clear the active conversation history (does not delete the persistent state.md).

/help — Display the list of commands.

/exit or /quit — Exit the application and display session statistics (time elapsed, tokens used, tool calls made).

Research & Security Basis

Conductor incorporates several modern LLM development patterns:

ReAct Pattern: Based on Yao et al. (2022) for alternating thought/action.

Prompt Injection Defense: Follows OWASP LLM Top 10 guidelines by explicitly wrapping web data as untrusted content.

Local RAG Chunking: Optimized 400-word chunks with 50-word overlaps using multi-qa-mpnet-base-dot-v1 embeddings (preferred over MiniLM for retrieval tasks).