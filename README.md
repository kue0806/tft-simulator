# TFT Simulator

A complete game simulator and real-time decision optimization tool for Teamfight Tactics (TFT) Set 16 "Lore & Legends".

## Project Structure

```
tft-simulator/
├── src/
│   ├── core/           # Core simulation modules
│   ├── optimizer/      # Optimization engine
│   ├── data/           # Data models and loaders
│   └── api/            # API endpoints
├── tests/              # Test suite
├── data/
│   ├── champions/      # Champion data
│   ├── items/          # Item data
│   └── traits/         # Trait data
├── config/             # Configuration files
└── docs/               # Documentation
```

## Setup

### Prerequisites
- Python 3.11+
- Conda (Miniconda or Anaconda)

### Installation

1. Create and activate the conda environment:
```bash
conda activate tft-sim
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

*Coming soon*

## Development

Run tests:
```bash
pytest
```

Start the API server:
```bash
uvicorn src.api.main:app --reload
```

## License

MIT
