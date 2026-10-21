# Logseq Lens

An agentic research assistant for your Logseq knowledge base. Combines traditional RAG with LangGraph-based agentic retrieval for deep research over your Markdown notes.

## Features

- **Hybrid Retrieval**: Keyword (SQLite FTS5) + Vector (Chroma) search with Reciprocal Rank Fusion
- **Agentic Research**: LangGraph agent that performs multi-step investigation when needed
- **Research Toggle**: Enable/disable deep research per question
- **Multi-Graph Support**: Switch between Logseq graphs with isolated indexes
- **MCP Integration**: Expose search, sync, and graph tools via Model Context Protocol
- **Provider Flexibility**: Local (Ollama) or cloud (OpenRouter, Google Gemini, OpenAI, Anthropic) LLMs
- **Streamlit UI**: Clean chat interface with source display and research traces
- **Index Management**: Incremental sync, change detection, full rebuild

## Quick Start

### Prerequisites
- Python 3.10+
- Ollama running locally (for local LLM)
- Logseq graph with Markdown files
- API keys for cloud providers (optional):
  - Google Gemini: `GOOGLE_API_KEY`
  - OpenAI: `OPENAI_API_KEY`
  - Anthropic: `ANTHROPIC_API_KEY`
  - OpenRouter: `OPENROUTER_API_KEY`

### Installation

```bash
git clone <repo>
cd Logseq_Lens
pip install -r requirements.txt
cp .env.example .env  # Edit with your paths
```

### Configuration

Edit `.env`:
```env
LOGSEQ_PATH=/home/youruser/Logseq

# Local LLM (Ollama)
OLLAMA_MODEL=gemma4:e4b

# Cloud Providers (choose one)
# OpenRouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openrouter/free

# Google Gemini
GOOGLE_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-1.5-pro

# OpenAI
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_MODEL=claude-3-haiku-20240307

# Research
MAX_RESEARCH_ITERATIONS=3
DEFAULT_TOP_K=5
```

### Run

```bash
# Terminal 1: Streamlit UI
streamlit run ui.py

# Terminal 2: MCP server (optional)
python -m mcp_server.server
```

Open http://localhost:8501

## Usage

### Basic Chat
Ask questions about your notes:
- "What materials do I need to level Yelan?"
- "Summarize my meeting notes from last week"
- "What games have I played and what did I think?"

### Research Toggle
- **OFF (default)**: Fast single-pass RAG
- **ON**: Agent performs multiple searches, reads full pages, follows links/tags

### Graph Switching
Use the sidebar to select a different Logseq graph. Each graph maintains separate indexes.

### Maintenance
- **Resync**: Incremental update (new/modified/deleted files)
- **Rebuild**: Full index rebuild

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                           │
│  Sidebar: Provider │ Research │ Graph │ Maintenance        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LangGraph: rewrite_query → retrieve → evaluate_context    │
│                           │              │                  │
│                           ▼              ▼                  │
│                      sufficient    insufficient            │
│                           │              │                  │
│                           ▼              ▼                  │
│                      generate ← research_agent (subgraph)  │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
      │   Chroma     │ │   SQLite    │ │  Logseq      │
      │   Vectors    │ │   FTS5      │ │  Filesystem  │
      └──────────────┘ └─────────────┘ └──────────────┘
```

## Research Agent Tools

| Tool | Purpose |
|------|---------|
| `hybrid_search` | Default RRF search (keyword + semantic) |
| `keyword_search` | Exact term matching |
| `vector_search` | Semantic similarity |
| `get_document` | Full chunk by stable_id |
| `get_full_page` | All chunks for a source file |
| `search_by_tag` | Pages with specific #tag |
| `search_by_page` | Pages by title |
| `find_related_pages` | Graph traversal (links, tags, backlinks) |
| `get_index_status` | Index diagnostics |
| `resync_graph` | Trigger sync |
| `check_graph_changes` | Detect modifications |

## MCP Server

Run `python -m mcp_server.server` to expose:

**Resources:**
- `logseq://graph/info` - Graph metadata
- `logseq://graph/index-status` - Index statistics
- `logseq://page/{stable_id}` - Page content
- `logseq://pages/by-tag/{tag}` - Pages by tag

**Tools:**
- `search_logseq`, `get_page`, `get_full_page`
- `get_related_pages`, `search_tags`
- `get_index_status`, `resync_graph`, `check_graph_changes`
- `set_graph_path`

Connect with any MCP client (Cursor, Claude Desktop, etc.)

## Project Structure

```
Logseq_Lens/
├── config.py                 # Configuration management
├── graph.py                  # Main LangGraph workflow
├── rag.py                    # Legacy compatibility layer
├── ui.py                     # Streamlit application
├── requirements.txt
├── .env                      # Your configuration (not committed)
├── services/
│   ├── retrieval.py          # Keyword/vector/hybrid search
│   ├── indexer.py            # Sync & change detection
│   ├── graph_config.py       # Per-graph index isolation
│   └── logseq_parser.py      # Links, tags, properties
├── tools/
│   └── research_tools.py     # LangGraph tools
├── agents/
│   ├── research_agent.py     # Agentic research subgraph
│   └── prompts.py            # Agent prompts
└── mcp_server/
    └── server.py             # FastMCP server
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LOGSEQ_PATH` | `~/Logseq` | Logseq graph directory |
| `OLLAMA_MODEL` | `gemma4:e4b` | Local model name |
| `OPENROUTER_MODEL` | `openrouter/free` | Cloud model name |
| `OPENROUTER_API_KEY` | - | Required for OpenRouter |
| `GOOGLE_API_KEY` | - | Required for Gemini |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Gemini model name |
| `OPENAI_API_KEY` | - | Required for OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `ANTHROPIC_API_KEY` | - | Required for Anthropic |
| `ANTHROPIC_MODEL` | `claude-3-haiku-20240307` | Anthropic model name |
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | Sentence transformer |
| `DEFAULT_TOP_K` | `5` | Retrieval results |
| `MAX_RESEARCH_ITERATIONS` | `3` | Agent iteration limit |
| `LANGFUSE_*` | - | Observability (optional) |

## Provider Selection

The UI supports 5 LLM providers:

| Provider | Type | Use Case |
|----------|------|----------|
| **Ollama** | Local | Free, private, runs on your machine |
| **OpenRouter** | Cloud | Free/paid models via single API |
| **Google Gemini** | Cloud | Strong reasoning, large context |
| **OpenAI** | Cloud | GPT-4o, GPT-4o-mini, etc. |
| **Anthropic** | Cloud | Claude 3.5 Sonnet/Haiku/Opus |

Select provider in the sidebar. Each provider has its own model dropdown with sensible defaults.

## Advanced Logseq Features

The parser extracts and indexes:
- **Page links**: `[[Page Name]]` for graph traversal
- **Tags**: `#tag` for topic-based search
- **Properties**: `key:: value` at page/block level
- **TODO states**: `TODO`, `DONE`, etc.
- **Journal dates**: Auto-detected from `journals/YYYY_MM_DD.md`
- **Block UUIDs**: For fine-grained retrieval

## Development

```bash
# Run tests (when added)
pytest tests/

# Type check
mypy .

# Format
ruff format .
```

## License

MIT