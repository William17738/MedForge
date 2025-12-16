# MedForge

**LLM-Powered Parallel Document Processing Pipeline**

A high-performance system for transforming educational materials into structured, AI-enhanced study resources. Built to process textbooks, exercises, and presentations through parallel pipelines with intelligent LLM routing and fault-tolerant execution.

> *"Built by a medical student who saw the problem firsthand — automating the tedious process of organizing study materials."*

---

## Features

| Feature | Description |
|---------|-------------|
| **Parallel Pipelines** | Three concurrent processing streams: exercises, key points, and lecture integration |
| **Multi-Provider LLM** | Unified interface for Gemini, Claude, and OpenAI with automatic failover |
| **Smart Routing** | Exponential backoff, quota management, and periodic primary model restoration |
| **Fault Tolerance** | Chapter and item-level checkpointing for seamless resume after interruption |
| **Atomic State** | Per-subject status tracking prevents race conditions in parallel processing |
| **Hash-Based Caching** | Skip unchanged sources to minimize redundant processing |

## Architecture

```
+-----------------------------------------------------------+
|                      Input Sources                        |
|        (Textbooks, Exercises, Presentations)              |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
|                      Preprocessing                        |
|    Chapter splitting - Content separation - Hashing       |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
|                 Three Parallel Pipelines                  |
|  +-------------+   +-------------+   +-----------------+  |
|  |  Exercises  |   | Key Points  |   | PPT Integration |  |
|  |  - Parse    |   | - Extract   |   | - Merge content |  |
|  |  - Solve    |   | - Summarize |   | - Integration   |  |
|  +-------------+   +-------------+   +-----------------+  |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
|                      Final Assembly                       |
|          Merge chapters into complete documents           |
+-----------------------------------------------------------+
```

## Quick Start

### Prerequisites

- Python 3.10+
- At least one LLM API key (Gemini, Anthropic, or OpenAI)

### Installation

```bash
git clone https://github.com/yourusername/MedForge.git
cd MedForge
pip install -r requirements.txt
```

### Configuration

Set your API keys:

```bash
export GEMINI_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
```

### Usage

```bash
# Process all subjects
python run_all.py

# Process a specific subject
python run_all.py Biology

# Run individual pipelines
python chapter_runner.py Biology
python qpoints_runner.py Biology
python ppt_runner.py Biology
```

## Demo (No API Key Required)

Try the offline demo to see the output format and pipeline structure:

```bash
python run_demo.py
```

**Output:**
```
demo/output/
|-- Biology_exercises_demo.md
|-- Biology_key_points_demo.md
+-- Biology_lecture_demo.md
```

For pre-generated LLM-powered output examples, see `demo/output_example/`.

## Privacy & Data

- **No patient data** - This project processes educational materials only
- **Demo uses synthetic samples** - English biology content under `demo/input/`
- **Production tested on real data** - Aggregated metrics below (no raw data included)

## Production Stats (Aggregated)

| Metric | Value |
|--------|-------|
| Questions processed | ~5,000+ |
| Subjects covered | 12 |
| Resume success rate | 98%+ |
| LLM routing | Gemini 85% / Claude 12% / OpenAI 3% |

_Stats from private Chinese medical study materials - raw data not included for copyright/privacy reasons._

## Technical Highlights

### Multi-Provider LLM Client

```python
# Unified interface for multiple providers
class LLMProvider(ABC):
    @abstractmethod
    def call(self, prompt: str, model: str) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

# Implementations for each provider
class GeminiProvider(LLMProvider): ...
class AnthropicProvider(LLMProvider): ...
class OpenAIProvider(LLMProvider): ...
```

### Smart Routing with Failover

```python
class ModelRouter:
    """Thread-safe model routing with exponential backoff."""

    def should_retry_primary(self) -> bool:
        # Checks: cooldown elapsed, request count, time since fallback

    def switch_to_fallback(self, model: str):
        # Atomic state update with lock protection
```

### Two-Level Parallelism

```
Process Pool (configurable, default 8)
+-- Thread Pool per process (configurable, default 4)
    +-- Total concurrency: 32 parallel LLM calls
```

### Atomic State Management

```python
class SubjectStatusManager:
    """Per-subject status files prevent race conditions."""

    def get_preprocess_status(self) -> dict
    def set_preprocess_status(self, source_hash: str, **metadata)
```

## Project Structure

```
MedForge/
|-- config.py              # Configuration and settings
|-- llm_client.py          # Multi-provider LLM abstraction
|-- status_manager.py      # Atomic state management
|-- preprocessor.py        # Document preprocessing
|-- run_all.py             # Main orchestrator
|-- run_demo.py            # Offline demo script
|-- brush_group.py         # Exercise processing pipeline
|-- qpoints_group.py       # Key points extraction pipeline
|-- ppt_group.py           # PPT integration pipeline
|-- final_assembler.py     # Document assembly
|-- demo/
|   |-- input/             # Sample English input files
|   +-- output_example/    # Pre-generated LLM output samples
+-- ARCHITECTURE.md        # Detailed architecture documentation
```

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `MEDFORGE_PROCESSES` | Number of parallel processes | 8 |
| `MEDFORGE_THREADS` | Threads per process | 4 |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |

## Output

For each subject, MedForge generates three documents:

1. **`{subject}_exercises_complete.md`** - Solved exercises with explanations
2. **`{subject}_key_points_complete.md`** - Key knowledge points summary
3. **`{subject}_lecture_notes_complete.md`** - Integrated lecture notes

## Design Philosophy

1. **Domain-Driven**: Built from real educational workflow needs
2. **Fault-First**: Every component designed for graceful failure handling
3. **Scale-Ready**: Two-level parallelism for resource-adaptive scaling
4. **Provider-Agnostic**: Easy to add new LLM providers
5. **Observable**: Comprehensive logging for debugging and monitoring

## Why I Built This

As a medical student, I spent countless hours organizing study materials — splitting textbooks, correlating exercises with content, and creating summary notes. This project automates that entire workflow, letting LLMs do the heavy lifting while maintaining the quality and structure needed for effective studying.

The technical challenges were fascinating:
- Managing concurrent LLM API calls without hitting rate limits
- Implementing fault-tolerant processing across thousands of questions
- Designing state management that survives process crashes
- Creating a provider abstraction that makes switching LLMs trivial

## License

MIT License - See [LICENSE](LICENSE) for details.

---

**Built with Python** | **Powered by Gemini, Claude, and GPT**
