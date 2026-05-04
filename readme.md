# NewsBot Intelligence System

A comprehensive NLP system for news article classification, analysis, and intelligent processing. Combines machine learning for text classification with advanced NLP features including topic modeling, sentiment analysis, entity extraction, and multilingual support.

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Testing](#testing)

## 🔧 How It Works

### System Architecture

The NewsBot system is built on a modular architecture with distinct layers:

```
Input (News Articles)
    ↓
Data Processing Layer
    ├─ Text Normalization (lowercase, punctuation removal)
    ├─ Text Combining (headline + content)
    └─ Validation & Filtering
    ↓
Training Pipeline
    ├─ TF-IDF Vectorization (converts text to numerical features)
    ├─ Logistic Regression (classification model)
    └─ Metric Evaluation (accuracy, F1 score)
    ↓
Advanced NLP Features
    ├─ Topic Modeling (NMF extracts topics)
    ├─ Sentiment Analysis (positive/negative words)
    ├─ Entity Extraction (identifies key entities)
    ├─ Summarization (intelligent text summaries)
    └─ Multilingual Processing (language detection & translation)
    ↓
Inference Service
    └─ API Server (FastAPI)
        ├─ Single & Batch Predictions
        ├─ Article Analysis
        ├─ Semantic Search
        └─ Health & Metrics Endpoints
```

### Key Components

#### 1. **Data Processing** (`src/data_processing/`)
- **text_preprocessor.py**: Normalizes text (removes punctuation, extra spaces, converts to lowercase)
- **feature_extractor.py**: Extracts keywords and features from text
- **data_validator.py**: Validates data quality and structure

#### 2. **NLP Pipeline** (`src/nlp_pipeline/`)
- **config.py**: Pydantic models for configuration management
- **data.py**: Loads and validates datasets (CSV/Parquet), combines text columns
- **preprocessing.py**: Regex-based text cleaning, tokenization, sentence splitting
- **modeling.py**: Builds sklearn pipeline with TF-IDF vectorizer + Logistic Regression
- **pipeline.py**: Orchestrates training workflow, saves artifacts
- **inference.py**: InferenceService for predictions and article analysis
- **api.py**: FastAPI application with prediction and analysis endpoints

#### 3. **Advanced NLP** (`src/nlp_pipeline/newsbot.py`)
A comprehensive `NewsBotSystem` class with multiple specialized components:

- **NewsTopicModeler**: Uses Non-negative Matrix Factorization (NMF) to extract topics
- **IntelligentSummarizer**: Extracts key sentences and creates summaries
- **NewsSentimentTracker**: Analyzes sentiment using positive/negative word lexicons
- **EntityRelationshipMapper**: Identifies and categorizes named entities
- **MultilingualProcessor**: Detects language and provides basic translation
- **SemanticSearchEngine**: Implements TF-IDF based similarity search
- **ConversationalInterface**: Enables chat-like interaction with the system

#### 4. **Monitoring** (`src/nlp_pipeline/monitoring.py`)
- Prometheus metrics tracking
- Request timing and latency monitoring
- Prediction distribution tracking
- Data drift detection

### Data Flow

1. **Training Phase**:
   - Load CSV data (train.csv, validation.csv)
   - Combine text columns (headline + content → single text field)
   - Normalize text (remove punctuation, lowercase)
   - Create TF-IDF features (20,000 max features, bigrams)
   - Train Logistic Regression classifier
   - Extract topics, sentiment, entities from training data
   - Save model artifacts and corpus

2. **Inference Phase**:
   - User submits text via API
   - Text is normalized
   - Model predicts category with confidence scores
   - Advanced NLP features analyze the text:
     - Extract topics using learned topic model
     - Calculate sentiment score
     - Identify entities
     - Generate summary
   - Return comprehensive analysis JSON

## 🚀 Quick Start

### Option 1: Using Python Directly

```bash
# 1. Run tests
python main.py test

# 2. Train the model
python main.py train

# 3. Start the API
python main.py serve
# API is now at http://localhost:8000
```

### Option 2: Using Docker (Recommended)

```bash
# Full setup: test, train, build, and run
python main.py setup

# Or step by step:
python main.py test
python main.py train
python main.py docker build
python main.py docker compose
```

## 📦 Installation

### Prerequisites

- Python 3.11+ (or 3.13.9 as configured)
- pip or conda
- Docker & Docker Compose (optional, for containerized deployment)

### Local Setup

```bash
# Clone or navigate to project
cd final-news-bot

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py --help
```

### Dependencies

- **fastapi** (0.116.1): Web framework for APIs
- **uvicorn** (0.35.0): ASGI server
- **pandas** (2.3.1): Data manipulation
- **scikit-learn** (1.7.0): ML models and NLP features
- **numpy** (2.2.6): Numerical computations
- **pydantic** (2.11.7): Data validation
- **PyYAML** (6.0.2): Configuration management
- **pytest** (8.4.1): Testing framework

## 💻 Usage

### Training the Model

```bash
# Train with default config
python main.py train

# Train with custom config
python main.py train --config path/to/config.yaml
```

**What happens during training:**
1. Loads data from `data/raw/train.csv` and `data/raw/validation.csv`
2. Validates and preprocesses text
3. Trains TF-IDF + Logistic Regression model
4. Evaluates on validation set
5. Saves model to `artifacts/model.joblib`
6. Saves metrics, classification report, corpus, and manifest

**Output:** `artifacts/manifest.json` with training summary

### Starting the API Server

```bash
# Default (0.0.0.0:8000)
python main.py serve

# Custom host and port
python main.py serve --host 127.0.0.1 --port 8080

# With auto-reload (development)
python main.py serve --reload
```

**API will be available at:**
- Swagger Docs: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`
- Health Check: `http://localhost:8000/health`

### Running Tests

```bash
python main.py test
```

**Test Coverage:**
- ✅ Text preprocessing (punctuation removal, normalization)
- ✅ Model training and prediction
- ✅ Pipeline orchestration
- ✅ Article analysis end-to-end
- ✅ Topic modeling
- ✅ Classification accuracy

## 🔗 API Endpoints

### Health & Status

```bash
# Health check
GET /health
# Response: {"status": "ok"}

# Prometheus metrics
GET /metrics
```

### Predictions

```bash
# Single prediction
POST /predict
Content-Type: application/json
{
  "text": "Apple announces new AI features in iOS 18"
}

# Response:
{
  "label": "technology",
  "confidence": 0.95,
  "probabilities": {
    "technology": 0.95,
    "business": 0.04,
    "science": 0.01,
    ...
  },
  "alternatives": [
    {"label": "business", "confidence": 0.04},
    {"label": "science", "confidence": 0.01}
  ]
}
```

```bash
# Batch predictions
POST /predict/batch
Content-Type: application/json
{
  "items": [
    {"text": "Article 1"},
    {"text": "Article 2"}
  ]
}
```

### Analysis

```bash
# Full article analysis (classification + topics + sentiment + entities)
POST /analyze
Content-Type: application/json
{
  "text": "article text here",
  "metadata": {
    "source": "BBC",
    "published_at": "2026-04-28"
  }
}

# Response includes:
{
  "classification": {
    "label": "technology",
    "confidence": 0.95
  },
  "topics": [
    {"topic_id": 0, "weight": 0.8, "keywords": ["ai", "apple", "software"]},
    ...
  ],
  "sentiment": {
    "score": 0.6,
    "label": "positive"
  },
  "entities": [
    {"text": "Apple", "type": "organization"},
    ...
  ],
  "summary": "Apple announces new AI features...",
  "language": "en"
}
```

### Search

```bash
# Semantic search over ingested articles
POST /search
Content-Type: application/json
{
  "query": "artificial intelligence",
  "top_k": 5
}

# Response:
{
  "query": "artificial intelligence",
  "results": [
    {"article": {...}, "relevance_score": 0.92},
    ...
  ]
}
```

## 🐳 Docker Deployment

### Build Docker Image

```bash
python main.py docker build
# Creates image: newsbot:latest
```

### Run with Docker Compose

```bash
python main.py docker compose
# Starts: API (port 8000), Prometheus (9090), Grafana (3000)
```

### Check Status

```bash
python main.py status
# Shows running containers and API health
```

### Push to Docker Hub

```bash
python main.py docker push yourusername
# Tags and pushes to Docker Hub registry
```

**docker-compose.yml** starts:
- **API**: NewsBot FastAPI service (port 8000)
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Dashboard visualization (port 3000)

## 📁 Project Structure

```
final-news-bot/
├── main.py                          # Unified CLI (test, train, serve, docker)
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container definition
├── docker-compose.yml               # Multi-container setup
├── configs/
│   └── pipeline.yaml               # Model and data configuration
├── data/
│   ├── raw/                        # Original datasets
│   │   ├── train.csv
│   │   └── validation.csv
│   ├── processed/                  # Processed data
│   └── models/                     # Saved models
├── artifacts/                      # Training outputs
│   ├── model.joblib               # Trained sklearn pipeline
│   ├── metrics.json               # Performance metrics
│   ├── classification_report.json
│   ├── drift_snapshot.json        # Data drift detection
│   ├── news_corpus.json           # Ingested articles
│   └── manifest.json              # Training summary
├── src/
│   ├── data_processing/           # Data utilities
│   ├── language_models/           # Embeddings, summarization
│   ├── multilingual/              # Language detection, translation
│   └── nlp_pipeline/              # Core system
│       ├── api.py                 # FastAPI app
│       ├── main.py                # CLI entrypoint
│       ├── config.py              # Configuration
│       ├── data.py                # Data loading
│       ├── preprocessing.py       # Text normalization
│       ├── modeling.py            # ML models
│       ├── inference.py           # Predictions
│       ├── newsbot.py             # Advanced NLP
│       ├── pipeline.py            # Training orchestration
│       ├── monitoring.py          # Metrics & monitoring
│       └── logging_utils.py       # Logging setup
├── tests/                         # Test suite
│   ├── test_classification.py
│   ├── test_preprocessing.py
│   ├── test_pipeline.py
│   ├── test_topic_modeling.py
│   └── test_integration.py
├── notebooks/                     # Jupyter notebooks for exploration
└── docs/                          # Documentation
```

## 🧪 Testing

### Run All Tests

```bash
python main.py test
```

### Run Specific Test

```bash
python -m pytest tests/test_classification.py -v
```

### Test Categories

| Test | Purpose |
|------|---------|
| `test_classification.py` | Verify NewsClassifier trains and predicts |
| `test_preprocessing.py` | Validate text normalization |
| `test_pipeline.py` | Test full training pipeline |
| `test_topic_modeling.py` | Verify topic extraction |
| `test_integration.py` | End-to-end newsbot flow |

**Current Status:** All 7 tests passing ✅

## ⚙️ Configuration

Edit `configs/pipeline.yaml` to customize:

```yaml
project_name: "newsbot-intelligence-system"
artifacts_dir: "artifacts"

data:
  train_path: "data/raw/train.csv"
  validation_path: "data/raw/validation.csv"
  text_column: "text"
  text_columns: ["headline", "content"]  # Combine these
  label_column: "category"
  min_text_length: 25
  allowed_labels:
    - business
    - politics
    - technology
    - science
    - sports

model:
  max_features: 20000          # TF-IDF max features
  ngram_range: [1, 2]         # Unigrams and bigrams
  classifier: "logistic_regression"
  random_state: 42

monitoring:
  service_name: "newsbot-intelligence-api"
  drift_sample_size: 1000
```

## 📊 Performance Metrics

Training on sample data achieves:
- **Accuracy**: 70%
- **Macro F1 Score**: 0.693

Metrics saved to: `artifacts/metrics.json`

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'nlp_pipeline'` | Run `python main.py` from project root with PYTHONPATH auto-setup |
| `Model artifact not found` | Run `python main.py train` first |
| Port 8000 already in use | Use `python main.py serve --port 8080` |
| Docker build fails | Ensure Docker daemon is running and data files exist |

## 📝 Example Usage Flow

```bash
# 1. Setup
cd final-news-bot
pip install -r requirements.txt

# 2. Validate with tests
python main.py test

# 3. Train model
python main.py train

# 4. Start API
python main.py serve

# 5. Test API (in another terminal)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "New smartphone with advanced AI features released"}'

# 6. Deploy with Docker
python main.py docker compose
```

## 🎯 Key Features

- ✅ **Text Classification**: Categorize news articles (5 categories)
- ✅ **Topic Modeling**: Automatically extract topics using NMF
- ✅ **Sentiment Analysis**: Determine positive/negative sentiment
- ✅ **Entity Extraction**: Identify key entities (people, organizations)
- ✅ **Summarization**: Generate intelligent text summaries
- ✅ **Multilingual**: Language detection and basic translation
- ✅ **Semantic Search**: Find similar articles in corpus
- ✅ **Batch Processing**: Handle multiple articles at once
- ✅ **Monitoring**: Prometheus metrics for performance tracking
- ✅ **Docker Ready**: Container deployment with docker-compose

## 📚 Additional Resources

- **API Docs**: http://localhost:8000/docs (interactive Swagger UI)
- **Config Reference**: See `configs/pipeline.yaml`
- **Training Reports**: Check `artifacts/classification_report.json`
- **Notebooks**: Explore `notebooks/` for data exploration examples

## 🤝 Contributing

To add new features or fix bugs:

1. Create a new branch
2. Make changes and add tests
3. Run `python main.py test` to verify
4. Submit pull request

## 📄 License

This project is part of the NewsBot Intelligence System.

---

**Last Updated**: April 2026  
**Version**: 1.0.0  
**Status**: Fully Functional ✅
