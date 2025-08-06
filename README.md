# Leveraging NLP and LLMs for Smart Legal Contracts in Accord Project Format

## Overview

This project presents an end-to-end pipeline for automatically transforming natural language contract clauses into executable Smart Legal Contracts (SLCs) using the open-source Accord Project framework. The system combines advanced Natural Language Processing (NLP) techniques with Large Language Models (LLMs) to bridge the gap between traditional legal contracts and automated contract execution.

## Key Features

- **Automated Clause Classification**: Domain-adapted LegalBERT for accurate contract clause categorization
- **Variable Extraction**: Hybrid rule-based and LLM-powered extraction for dates, amounts, parties, and durations
- **Template Generation**: Complete Accord Project-compliant artifacts (CiceroMark templates, Concerto models, TypeScript logic)
- **State Machine Generation**: Intelligent FSM creation with validation
- **Knowledge Graph Integration**: Neo4j-based contract relationship modeling
- **Privacy-Preserving**: Local processing with redaction detection
- **Scalable Architecture**: Modular pipeline supporting batch and real-time processing

## Installation

### Prerequisites

- **Python 3.8+** (recommended: Python 3.10+)
- **Node.js 16+** (for Accord Project templates)
- **Neo4j 4.4+** (optional, for knowledge graph features)
- **Git** (for version control)
- **8GB+ RAM** (recommended for LLM processing)

### Setup

```bash
# Clone the repository
git clone https://github.com/3TH4N5/smart-legal-contracts-nlp.git
cd smart-legal-contracts-nlp

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Windows CMD:
.venv\Scripts\activate.bat
# On Linux/Mac:
source .venv/bin/activate

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install core dependencies (this may take 5-10 minutes)
pip install -r requirements.txt

# Install Accord Project CLI (requires Node.js)
npm install -g @accordproject/cicero-cli

# Download CUAD dataset and setup directories
python scripts/download_cuad.py

# Download and setup NLP models
python scripts/download_models.py

# Verify installation
python -c "import torch, transformers, ollama; print('✓ Core libraries installed')"
```

### Initial Data Processing

After setup, run these commands to build up the pipeline components:

```bash
# 1. Preprocess CUAD dataset
python -m src.preprocessing.cuad_preprocessor

# 2. Train binary classifier (answer detection)
python -m src.classification.binary_classifier

# 3. Train multiclass classifier (clause categorization)
python -m src.classification.multiclass_classifier

# 4. Generate clause embeddings for similarity matching
python -m src.embeddings.clause_embedder

# 5. Build similarity inference engine
python -m src.inference.similarity_engine

# 6. Run comprehensive evaluation
python -m docs.evaluation.comprehensive_evaluator
```

### Dependencies Overview

The project uses 100+ carefully selected packages organized by category:

**Core ML/NLP:**

- PyTorch, Transformers, Sentence-Transformers
- SpaCy, NLTK, TextBlob for text processing

**Data Processing:**

- Pandas, NumPy, Scikit-learn
- PyArrow for efficient data handling

**LLM Integration:**

- Ollama for local LLM processing
- OpenAI and Anthropic APIs (optional)

**Template Generation:**

- Jinja2, Markdown, LXML
- Custom Accord Project integration

### Configuration Files

The system uses JSON configuration files (no `.env` required):

```bash
config/
├── variable_targets.json         # 200+ variables across 41 clause types
├── template_mapping.json         # Accord Project template mapping
├── state_machine_config.json     # FSM generation & validation
├── graph_config.json            # Neo4j settings (optional)
└── knowledge_extraction.json    # Knowledge graph extraction rules
```

### Local Model Setup

```bash
# Install Ollama
# Visit: https://ollama.ai/download

# Pull required model
ollama pull llama3
```

### Dataset Download

The project uses the **Contract Understanding Atticus Dataset (CUAD)**:

```bash
# Automatic download (recommended)
python scripts/download_cuad.py

# Manual download if automatic fails:
# 1. Visit: https://github.com/TheAtticusProject/cuad
# 2. Download data.zip
# 3. Extract to: data/raw/cuad/
```

The download script will:

- Create proper directory structure
- Download from official CUAD GitHub repository
- Extract and organize dataset files
- Verify data integrity and structure

### Neo4j Setup (Optional)

For knowledge graph functionality:

#### Install Neo4j Community Edition

**Download and Install:**

- Visit: https://neo4j.com/download/
- Download Neo4j Desktop or Neo4j Community Server
- Follow installation instructions for your platform

**Configure Neo4j:**

```bash
# Default configuration in config/graph_config.json:
{
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "password",
    "database": "neo4j"
  }
}
```

**Set Password:**

1. Start Neo4j service
2. Open browser to http://localhost:7474
3. Login with username: `neo4j`, password: `neo4j`
4. Set new password to `password` (or update config file accordingly)

**Install Python Driver:**

```bash
pip install neo4j python-slugify
```

## Quick Start

### Complete Pipeline (Recommended)

```bash
# Use mode defaults
python -m src.fsm.enhanced_pipeline --mode test      # 5 contracts, ~2min
python -m src.fsm.enhanced_pipeline --mode default   # 15 contracts, ~5min
python -m src.fsm.enhanced_pipeline --mode production # 100 contracts, ~30min

# Override limits (can scale to 510 contracts)
python -m src.fsm.enhanced_pipeline --mode test --max-contracts 25
python -m src.fsm.enhanced_pipeline --mode production --max-contracts 510  # Full dataset

# Skip components
python -m src.fsm.enhanced_pipeline --mode default --no-templates
python -m src.fsm.enhanced_pipeline --variables-only --max-contracts 10
```

### Individual Components

#### Variable Extraction

```bash
# Use mode defaults
python -m src.extraction.variable_extractor --mode test        # 5 contracts
python -m src.extraction.variable_extractor --mode production  # 100 contracts

# Override limits
python -m src.extraction.variable_extractor --mode test --contracts 25
python -m src.extraction.variable_extractor --mode production --contracts 510  # Full dataset
```

#### Template Generation

```bash
# Basic generation (requires extraction results)
python -m src.generation.template_generator \
  --extraction-file outputs/extracted_variables/latest.json

# With contract limits
python -m src.generation.template_generator \
  --extraction-file data.json \
  --max-contracts 50 \
  --force

# Full dataset
python -m src.generation.template_generator \
  --extraction-file data.json \
  --max-contracts 510
```

#### Knowledge Graph Construction

```bash
# Build knowledge graph from extractions
python -m src.knowledge_graph.graph_builder --max-contracts 30 --clear --verbose

# Auto-detect latest extraction file
python -m src.knowledge_graph.graph_builder --clear

# Use specific extraction file
python -m src.knowledge_graph.graph_builder \
  --extraction-file outputs/extracted_variables/extracted_variables_20250101_120000.json \
  --max-contracts 50 \
  --clear
```

#### Pipeline Benchmarking

```bash
# Use mode defaults for performance analysis
python -m src.benchmarking.pipeline_benchmarker --mode test       # 5 contracts
python -m src.benchmarking.pipeline_benchmarker --mode production # 100 contracts

# Override limits
python -m src.benchmarking.pipeline_benchmarker --mode test --contracts 25
python -m src.benchmarking.pipeline_benchmarker --mode production --contracts 510  # Full dataset
```

## Dataset

The project uses the **Contract Understanding Atticus Dataset (CUAD)**:

- **Size**: 13,000+ annotated legal clauses from 510 commercial contracts
- **Source**: SEC filings with expert legal annotations
- **Categories**: 41 distinct clause types
- **Download**: Automated via `python scripts/download_cuad.py`
- **Processing**: Automatic discovery and normalization of contract data

### CUAD Download Script Features

The `scripts/download_cuad.py` script provides:

```python
def download_cuad():
    """Download CUAD dataset from GitHub with fallback URLs"""
    # Tries multiple GitHub URLs for reliability
    # Creates proper directory structure
    # Handles extraction and cleanup

def explore_cuad():
    """Explore downloaded CUAD data structure"""
    # Analyzes file types and sizes
    # Validates data integrity
    # Provides data overview statistics
```

**Manual Download Fallback:**
If automatic download fails, the script provides clear instructions for manual setup from the official CUAD repository.

## System Architecture

The pipeline consists of five main stages:

1. **Input Processing**: Contract text normalization and clause extraction
2. **Classification**: Two-stage legal clause categorization using LegalBERT
3. **Variable Extraction**: Hybrid rule-based and LLM-powered variable identification
4. **Template Generation**: Accord Project artifact synthesis with unified Double type system
5. **State Machine Generation**: Intelligent FSM creation with formal validation
6. **Knowledge Graph Construction**: Semantic relationship modeling (optional)

## Performance

### Classification Performance

- **Two-stage pipeline**: Binary filtering followed by multi-class categorization
- **LegalBERT integration**: Domain-adapted transformer model for legal text
- **Intelligent processing**: Efficient data filtering and high-precision clause identification

### Variable Extraction Results

- **DateTime**: High accuracy extraction with ISO 8601 normalization
- **MonetaryAmount**: Robust currency detection and normalization
- **Duration**: Enhanced pattern recognition for temporal expressions
- **Party**: Multi-method LLM extraction for entity identification
- **Double**: Unified numeric system handling integers, decimals, and percentages

### Template Generation Quality

- **Accord Project Compliance**: Full compatibility with official framework standards
- **Original Text Preservation**: Maintains authentic legal language in generated templates
- **CiceroMark Syntax**: Proper {{variableName}} placeholder implementation
- **Runtime Executability**: Generated templates successfully instantiate and execute

### State Machine Generation

- **FSM Validation**: Formal finite state machine validation with reachability analysis
- **Robust JSON Parsing**: Intelligent parsing with multiple fallback strategies
- **State Classification**: Automatic categorization into lifecycle phases
- **Confidence Tracking**: Per-state and per-transition confidence metrics

## Dynamic Contract Limits

The system uses intelligent mode-based contract limits to balance processing time and resource usage:

| Mode         | Contracts | Use Case          | Time\* | Override        |
| ------------ | --------- | ----------------- | ------ | --------------- |
| `debug`      | 1         | Minimal debugging | ~30s   | `--contracts N` |
| `test`       | 5         | Quick validation  | ~2min  | `--contracts N` |
| `default`    | 15        | Balanced testing  | ~5min  | `--contracts N` |
| `production` | 100       | Full analysis     | ~30min | `--contracts N` |

\*Approximate time for complete pipeline. **All components can scale up to 510 contracts (full CUAD dataset).**

**Why these limits?** Optimized for LLM processing speed, memory usage, and caching efficiency while providing meaningful results.

## Project Structure

```
smart-legal-contracts/
├── .venv/                                 # Virtual environment
├── cache/                                 # Cached data and models
├── config/                               # Configuration files
├── data/                                 # Raw and processed datasets
├── docs/                                 # Documentation
│   ├── evaluation/                       # Model evaluation results
│   └── evaluation_report/                # Detailed evaluation reports
├── models/                               # Trained models
├── outputs/                              # Generated results
│   ├── benchmarks/                       # Performance reports
│   ├── extracted_variables/              # Variable extraction output
│   ├── generated_templates/              # Accord Project templates
│   ├── reports/                          # Analysis reports
│   └── state_machines/                   # FSM definitions with validation
├── scripts/                              # Utility scripts
├── src/                                  # Source code
│   ├── __pycache__/                      # Python cache
│   ├── benchmarking/                     # Performance analysis
│   ├── classification/                   # Clause classification
│   ├── embeddings/                       # Vector embeddings
│   ├── extraction/                       # Variable extraction
│   ├── fsm/                             # State machine generation
│   ├── generation/                       # Template generation
│   ├── inference/                        # Model inference
│   ├── knowledge/                        # Knowledge graph
│   ├── preprocessing/                    # Data preprocessing
│   └── validation/                       # Validation utilities
├── utils/                                # Utility functions
├── .env                                  # Environment variables
├── .gitignore                           # Git ignore rules
├── README.md                            # Project documentation
└── requirements.txt                     # Python dependencies
```

## Testing

```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# End-to-end pipeline tests
python -m pytest tests/e2e/

# Component-specific tests
python -m pytest tests/extraction/
python -m pytest tests/generation/
python -m pytest tests/fsm/
```

## Output Structure

### Variable Extraction Output

```json
{
  "metadata": {
    "mode": "default",
    "max_contracts": 15,
    "total_extractions": 87,
    "has_legalbert": true
  },
  "extractions": [
    {
      "contract_id": "contract_001",
      "clause_type": "parties",
      "extractions": {
        "parties": [
          {
            "$class": "org.accordproject.organization.Organization",
            "partyId": "Apple Inc."
          }
        ]
      },
      "original_text": "Agreement between Apple Inc. and Microsoft Corporation..."
    }
  ]
}
```

### Template Generation Output

Generated templates include:

- `text/grammar.tem.md` - CiceroMark template with {{variableName}} placeholders
- `model/model.cto` - Concerto data model with unified Double type system
- `logic/logic.ts` - TypeScript business logic
- `request.json` - Sample template data
- `state.json` - Contract state management
- `package.json` - Accord Project package configuration

### State Machine Output

- `state_machine.json` - XState configuration with FSM validation
- `diagram.mermaid` - Visual state diagram
- `summary.json` - Generation metrics and validation results

## Knowledge Graph Features

### Building the Knowledge Graph

```bash
# Build from latest extraction data
python -m src.knowledge_graph.graph_builder --clear --verbose

# Limit contracts and specify file
python -m src.knowledge_graph.graph_builder \
  --extraction-file outputs/extracted_variables/latest.json \
  --max-contracts 50 \
  --clear
```

### Neo4j Access

After running the knowledge graph builder:

**Web Interface:**

- URL: http://localhost:7474
- Username: `neo4j`
- Password: `password` (as configured)

**Connection String:**

- URI: `bolt://localhost:7687`
- Database: `neo4j`

### Sample Knowledge Graph Queries

```cypher
// Find liability limitation clauses with monetary caps
MATCH (c:Contract)-[:CONTAINS]->(cl:Clause {clause_type: "cap_on_liability"})
      -[:HAS_VARIABLE]->(v:Variable {var_type: "MonetaryAmount"})
WHERE v.value > "1000000"
RETURN c.title, cl.original_text, v.value

// Analyze template generation success by clause type
MATCH (cl:Clause)-[:GENERATES]->(t:Template)
WITH cl.clause_type as clause_type, count(cl) as clauses_with_templates
MATCH (all_cl:Clause {clause_type: clause_type})
WITH clause_type, clauses_with_templates, count(all_cl) as total_clauses
RETURN clause_type, total_clauses, clauses_with_templates,
       round(100.0 * clauses_with_templates / total_clauses) as success_rate
ORDER BY success_rate DESC

// Find all data for a specific contract
MATCH (c:Contract {contract_id: "ENERGOUSCORP_03_16_2017-EX-10.24-STRATEGIC_ALLIANCE_AGREEMENT"})
OPTIONAL MATCH (c)-[r1:CONTAINS]->(cl:Clause)
OPTIONAL MATCH (c)-[r2:INVOLVES]->(p:Party)
OPTIONAL MATCH (cl)-[r3:HAS_VARIABLE]->(v:Variable)
OPTIONAL MATCH (cl)-[r4:GENERATES]->(t:Template)
OPTIONAL MATCH (t)-[r5:USES_VARIABLES]->(tv:Variable)
RETURN c, r1, cl, r2, p, r3, v, r4, t, r5, tv
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatting
black src/ && isort src/

# Run linting
flake8 src/ && mypy src/
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Privacy and Ethics

- **Local Processing**: All sensitive data processed locally using Ollama
- **Redaction Detection**: Automatic identification and skipping of redacted content
- **Audit Trails**: Complete logging of all automated decisions with confidence tracking
- **Human-in-the-Loop**: Expert validation for critical contract provisions
- **Transparent AI**: Full documentation of GenAI usage with ethical guidelines

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{smartlegalcontracts2025,
  title={Leveraging NLP and LLMs for Smart Legal Contracts in Accord Project Format},
  author={[Your Name]},
  year={2025},
  school={University College London},
  type={MSc Business Analytics Dissertation}
}
```

## Acknowledgments

- **Accord Project** for the open-source smart contract framework
- **CUAD Dataset** creators for the comprehensive legal contract annotations
- **Hugging Face** for the transformer model infrastructure and LegalBERT
- **Ollama** for local LLM deployment
- **Neo4j** for graph database capabilities
- UCL School of Management supervisors and the Business Analytics program

## Support

- **Issues**: [GitHub Issues](https://github.com/3TH4N5/smart-legal-contracts-nlp/issues)

## Recent Updates

### v1.2.0 - Enhanced Pipeline

- **Unified Double Type System**: All numeric values (integers, decimals, percentages) use Double type
- **Dynamic Contract Limits**: Mode-based processing (test=5, default=15, production=100, debug=1)
- **FSM Validation**: Formal finite state machine validation with reachability analysis
- **Robust JSON Parsing**: Enhanced LLM response parsing with intelligent fallbacks
- **Comprehensive Benchmarking**: Academic-grade performance analysis and reporting
- **Knowledge Graph Integration**: Full Neo4j support with semantic relationship modeling

## Troubleshooting

### Common Issues

**Neo4j Connection Issues:**

```bash
# Check if Neo4j is running
neo4j status

# Start Neo4j service
neo4j start

# Check configuration
cat config/graph_config.json
```

**Ollama Model Issues:**

```bash
# Check if Ollama is running
ollama list

# Pull model if missing
ollama pull llama3
```

**Python Package Issues:**

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Install specific Neo4j packages
pip install neo4j python-slugify
```

**Note**: The system is designed to work out-of-the-box with local configurations and doesn't require environment variables. All processing happens locally for privacy and security.
