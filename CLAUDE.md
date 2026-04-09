# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered personal finance coaching application that tracks expenses, analyzes spending patterns, provides budget recommendations, and uses ML to predict savings.

## Project Structure

```
├── main.py              # Core logic: data layer, analysis engine, CLI interface
├── model.py             # ML model: Ridge regression for savings prediction
├── app.py               # Streamlit web UI dashboard
├── finance_data.csv     # Persistent storage for expense records
├── requirements.txt     # Python dependencies
└── savings_model.joblib # Trained ML model (generated after training)
```

## Commands

- **CLI Interface**: `python main.py`
- **Web Dashboard**: `streamlit run app.py`
- **Install Dependencies**: `pip install -r requirements.txt`

## Architecture

### Data Layer (`main.py` - `FinanceData`)
- CSV-based persistence with columns: date, income, food, rent, travel, shopping, utilities, entertainment, healthcare, other, savings, total_expense
- Automatic CSV initialization on first run

### Analysis Engine (`main.py` - `FinancialAnalyzer`)
- Categorizes expenses into **needs** (rent, utilities, healthcare, food, travel) and **wants** (shopping, entertainment, other)
- Detects overspending using 50/30/20 rule thresholds
- Calculates savings trend (improving/declining/stable)

### Budget System (`main.py` - `BudgetRecommender`)
- Implements 50/30/20 rule: 50% needs, 30% wants, 20% savings
- Compares actual spending to recommended budgets

### ML Model (`model.py` - `SavingsPredictor`)
- Uses Ridge regression (alpha=1.0) trained on historical data
- Features: income + 8 expense categories
- Auto-trains when ≥3 records exist, falls back to simple calculation otherwise
- Feature importance shows which expenses most affect savings

### UI (`app.py`)
- Streamlit dashboard with 4 pages: Dashboard, Add Expense, History, AI Predictions
- Interactive charts: pie chart (expense distribution), line chart (savings trend), bar chart (budget comparison)
- Sidebar quick-entry form for adding expenses

## Key Patterns

- `FinanceCoach` is the main facade coordinating all subsystems
- Records stored as dicts with string values from CSV, converted to float on load via `get_records_asFloats()`
- ML model lazy-loads: `_load_or_train()` checks for saved model before training
