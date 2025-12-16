# MedForge Architecture

## Overview

MedForge is an LLM-powered parallel document processing pipeline designed to transform raw educational materials into structured, enhanced study resources. The system processes textbooks, exercises, and presentation materials through three parallel pipelines, leveraging multiple LLM providers with intelligent failover.

## Key Features

- **Three-Pipeline Parallel Processing**: Exercise analysis, key points extraction, and lecture integration run concurrently
- **Two-Level Concurrency**: Configurable process × thread parallelism (default: 8 processes × 4 threads)
- **Smart LLM Routing**: Automatic failover across Gemini, Claude, and OpenAI with exponential backoff
- **Atomic State Management**: Per-subject status tracking prevents race conditions
- **Checkpoint Resume**: Chapter and item-level caching for fault-tolerant processing
- **Provider Abstraction**: Unified interface supporting multiple LLM backends

## System Architecture

```
+------------------------------------------------------------------+
|                            MedForge                              |
+------------------------------------------------------------------+
|                                                                  |
|  +--------------+      +----------------------------------+      |
|  |   Sources    |      |          Preprocessing           |      |
|  |  ----------  | ---> |  - Chapter splitting             |      |
|  |  Textbooks   |      |  - Content/Exercise separation   |      |
|  |  Exercises   |      |  - Hash-based skip detection     |      |
|  |  PPT Files   |      +----------------+-----------------+      |
|  +--------------+                       |                        |
|                                         v                        |
|  +------------------------------------------------------------+  |
|  |                 Three Parallel Pipelines                   |  |
|  |  +------------+   +------------+   +--------------------+  |  |
|  |  | Exercises  |   | Key Points |   |  PPT Integration   |  |  |
|  |  | Pipeline   |   | Pipeline   |   |  Pipeline          |  |  |
|  |  |            |   |            |   |                    |  |  |
|  |  | - Parsing  |   | - Extract  |   | - Merge PPT +      |  |  |
|  |  | - LLM solve|   |   from Q&A |   |   textbook content |  |  |
|  |  | - Caching  |   | - Summarize|   | - Deep integration |  |  |
|  |  +-----+------+   +-----+------+   +---------+----------+  |  |
|  +--------|--------------|--------------------|---------------+  |
|           |              |                    |                  |
|           v              v                    v                  |
|  +------------------------------------------------------------+  |
|  |                      Final Assembly                        |  |
|  |  - Merge chapters into complete documents                  |  |
|  |  - Generate three outputs per subject                      |  |
|  +------------------------------------------------------------+  |
|                                                                  |
+------------------------------------------------------------------+
```

## Core Components

### Configuration (`config.py`)
- Global settings for directories, concurrency, and model configuration
- Environment-based configuration for API keys
- Fallback model priority chain

### LLM Client (`llm_client.py`)
- **Provider Abstraction**: `LLMProvider` base class with implementations for Gemini, Claude, OpenAI
- **Smart Router**: `ModelRouter` class managing failover with exponential backoff
- **Thread Safety**: Lock-protected state for concurrent access

### State Management (`status_manager.py`)
- Per-subject status files in `output/<subject>/.status/`
- Hash-based change detection for skip optimization
- Atomic updates prevent race conditions in parallel processing

### Preprocessor (`preprocessor.py`)
- Chapter detection and splitting using regex patterns
- Content/exercise separation for study guides
- Configuration-based or auto-detection modes

### Pipeline Processors
- **`brush_group.py`**: Exercise parsing and LLM-based solution generation
- **`qpoints_group.py`**: Key points extraction from processed exercises
- **`ppt_group.py`**: PPT content integration with textbook material

### Orchestration (`run_all.py`)
- Coordinates preprocessing and three parallel pipelines
- Subject discovery from config or directory scan
- Process pool management for parallel execution

### Assembly (`final_assembler.py`)
- Merges chapter outputs into complete documents
- Produces three document types per subject

## Data Flow

```
1. Input Sources
   +-- Textbooks, Exercises, PPT files

2. Preprocessing
   |-- Split into chapters
   |-- Separate content and exercises
   +-- Store in output/<subject>/raw/

3. Parallel Processing
   |-- Pipeline 1: Exercise Processing
   |   |-- Parse questions (parser_ocr_questions.py)
   |   |-- Structure as JSON (questions_structured/)
   |   +-- Generate solutions via LLM (brush_group.py)
   |
   |-- Pipeline 2: Key Points
   |   +-- Extract knowledge points from exercises (qpoints_group.py)
   |
   +-- Pipeline 3: PPT Integration
       +-- Merge PPT with textbook content (ppt_group.py)

4. Assembly
   +-- Merge chapters into complete documents

5. Output
   |-- {subject}_exercises_complete.md
   |-- {subject}_key_points_complete.md
   +-- {subject}_lecture_notes_complete.md
```

## Concurrency Model

### Two-Level Parallelism

```
Level 1: Process Pool (NUM_PROCESSES)
|-- Process 1: Subject A, Chapter 1
|-- Process 2: Subject A, Chapter 2
|-- Process 3: Subject B, Chapter 1
+-- ...

Level 2: Thread Pool (THREADS_PER_PROCESS)
+-- Within each process, questions processed in parallel
```

### Work Stealing
The exercise pipeline implements global work stealing for load balancing across chapters with varying question counts.

## LLM Routing Strategy

```
Primary Model (e.g., Gemini)
    |
    v (on failure)
Retry with exponential backoff (up to RETRIES_PER_MODEL)
    |
    v (quota exhausted)
Switch to Fallback 1 (e.g., Claude)
    |
    v (on failure)
Switch to Fallback 2 (e.g., OpenAI)
    |
    v (periodic)
Attempt to restore Primary (after cooldown)
```

## Directory Structure

```
MedForge/
|-- config.py                 # Global configuration
|-- llm_client.py             # LLM provider abstraction
|-- status_manager.py         # Atomic state management
|-- preprocessor.py           # Document preprocessing
|-- parser_ocr_questions.py   # Question parsing
|-- brush_group.py            # Exercise pipeline
|-- qpoints_group.py          # Key points pipeline
|-- ppt_group.py              # PPT integration pipeline
|-- final_assembler.py        # Document assembly
|-- run_all.py                # Main orchestrator
+-- output/
    +-- <subject>/
        |-- raw/                    # Preprocessed chapters
        |-- questions_structured/   # Parsed questions (JSON)
        |-- chapters/               # Processed chapter outputs
        |-- .status/                # Status tracking files
        +-- *_complete.md           # Final assembled documents
```

## Configuration

### Environment Variables

```bash
# Required: At least one LLM provider
export GEMINI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Optional: Concurrency tuning
export MEDFORGE_PROCESSES=8
export MEDFORGE_THREADS=4
```

### Subject Configuration

Create `output/subject_config.json`:

```json
{
  "Biology": {
    "textbook": "biology_textbook.txt",
    "exercises": "biology_exercises.txt"
  }
}
```

## Usage

```bash
# Process all subjects
python run_all.py

# Process specific subject
python run_all.py Biology

# Run individual pipelines
python chapter_runner.py Biology
python qpoints_runner.py Biology
python ppt_runner.py Biology
```

## Design Principles

1. **Fault Tolerance**: Every component supports checkpoint resume
2. **Scalability**: Two-level parallelism scales with available resources
3. **Flexibility**: Provider abstraction allows easy addition of new LLMs
4. **Observability**: Comprehensive logging at each processing stage
5. **Efficiency**: Hash-based skip detection avoids redundant processing
