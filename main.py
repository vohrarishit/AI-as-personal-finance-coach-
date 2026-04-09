"""
Personal Finance Coach - Core Logic
Handles expense tracking, financial analysis, and AI-based insights.
"""

import csv
import os
from datetime import datetime
from typing import Optional
import numpy as np


class FinanceData:
    """Manages finance data storage and retrieval."""

    CSV_PATH = "finance_data.csv"
    COLUMNS = ["date", "income", "food", "rent", "travel", "shopping",
               "utilities", "entertainment", "healthcare", "other", "savings", "total_expense"]

    @classmethod
    def initialize_csv(cls):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(cls.CSV_PATH):
            with open(cls.CSV_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(cls.COLUMNS)

    @classmethod
    def add_record(cls, income: float, expenses: dict) -> dict:
        """Add a new financial record and return the calculated values."""
        total_expense = sum(expenses.values())
        savings = income - total_expense
        date = datetime.now().strftime("%Y-%m-%d")

        record = {
            "date": date,
            "income": income,
            **expenses,
            "savings": savings,
            "total_expense": total_expense
        }

        with open(cls.CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cls.COLUMNS)
            writer.writerow(record)

        return record

    @classmethod
    def get_all_records(cls) -> list:
        """Retrieve all financial records."""
        cls.initialize_csv()
        records = []
        with open(cls.CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        return records

    @classmethod
    def get_records_asFloats(cls) -> list:
        """Retrieve records with numeric values converted to float."""
        records = cls.get_all_records()
        for r in records:
            for key in r:
                if key != "date":
                    r[key] = float(r[key])
        return records


class FinancialAnalyzer:
    """Analyzes financial data and provides insights."""

    NEEDS_CATEGORIES = ["rent", "utilities", "healthcare", "food", "travel"]
    WANTS_CATEGORIES = ["shopping", "entertainment", "other"]

    def __init__(self, records: list):
        self.records = records

    def calculate_summary(self) -> dict:
        """Calculate summary statistics from all records."""
        if not self.records:
            return {}

        total_income = sum(r["income"] for r in self.records)
        total_expenses = sum(r["total_expense"] for r in self.records)
        total_savings = sum(r["savings"] for r in self.records)
        avg_income = total_income / len(self.records)
        avg_expenses = total_expenses / len(self.records)
        avg_savings = total_savings / len(self.records)

        return {
            "total_income": total_income,
            "total_expenses": total_expenses,
            "total_savings": total_savings,
            "avg_monthly_income": avg_income,
            "avg_monthly_expenses": avg_expenses,
            "avg_monthly_savings": avg_savings,
            "savings_rate": (total_savings / total_income * 100) if total_income > 0 else 0,
            "record_count": len(self.records)
        }

    def categorize_expenses(self, record: dict) -> dict:
        """Categorize expenses into needs vs wants."""
        needs = sum(record.get(cat, 0) for cat in self.NEEDS_CATEGORIES)
        wants = sum(record.get(cat, 0) for cat in self.WANTS_CATEGORIES)
        return {"needs": needs, "wants": wants}

    def detect_overspending(self, record: dict) -> list:
        """Detect categories where user is overspending."""
        warnings = []
        income = record["income"]

        # 50/30/20 rule thresholds
        needs_limit = income * 0.50
        wants_limit = income * 0.30
        savings_target = income * 0.20

        categorized = self.categorize_expenses(record)

        if categorized["needs"] > needs_limit:
            warnings.append(f"Needs spending (${categorized['needs']:.2f}) exceeds 50% of income (${needs_limit:.2f})")

        if categorized["wants"] > wants_limit:
            warnings.append(f"Wants spending (${categorized['wants']:.2f}) exceeds 30% of income (${wants_limit:.2f})")

        if record["savings"] < savings_target:
            warnings.append(f"Savings (${record['savings']:.2f}) below 20% target (${savings_target:.2f})")

        return warnings

    def get_savings_trend(self) -> dict:
        """Calculate savings trend over time."""
        if len(self.records) < 2:
            return {"trend": "insufficient_data"}

        savings = [r["savings"] for r in self.records]
        if len(savings) >= 3:
            recent_avg = np.mean(savings[-3:])
            older_avg = np.mean(savings[:-3]) if len(savings) > 3 else np.mean(savings[:-1])
        else:
            recent_avg = np.mean(savings)
            older_avg = savings[0]

        if recent_avg > older_avg * 1.1:
            trend = "improving"
        elif recent_avg < older_avg * 0.9:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "recent_avg": recent_avg,
            "older_avg": older_avg,
            "change_pct": ((recent_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0
        }

    def get_category_averages(self) -> dict:
        """Calculate average spending per category."""
        if not self.records:
            return {}

        categories = ["food", "rent", "travel", "shopping", "utilities", "entertainment", "healthcare", "other"]
        averages = {}

        for cat in categories:
            values = [r.get(cat, 0) for r in self.records]
            averages[cat] = sum(values) / len(values)

        return averages


class BudgetRecommender:
    """Provides budget recommendations based on income."""

    def __init__(self, income: float):
        self.income = income
        self.needs_budget = income * 0.50
        self.wants_budget = income * 0.30
        self.savings_target = income * 0.20

    def get_recommendations(self) -> dict:
        """Return recommended budgets for each category."""
        return {
            "needs_budget": self.needs_budget,
            "wants_budget": self.wants_budget,
            "savings_target": self.savings_target,
            "total_budget": self.needs_budget + self.wants_budget + self.savings_target
        }

    def compare_to_actual(self, record: dict) -> dict:
        """Compare actual spending to recommended budgets."""
        categorized = FinancialAnalyzer([]).categorize_expenses(record)

        return {
            "needs_diff": categorized["needs"] - self.needs_budget,
            "wants_diff": categorized["wants"] - self.wants_budget,
            "savings_diff": record["savings"] - self.savings_target,
            "needs_status": "over" if categorized["needs"] > self.needs_budget else "under",
            "wants_status": "over" if categorized["wants"] > self.wants_budget else "under",
            "savings_status": "below_target" if record["savings"] < self.savings_target else "on_track"
        }


class FinanceCoach:
    """Main facade class that coordinates all finance operations."""

    def __init__(self):
        FinanceData.initialize_csv()
        self.records = FinanceData.get_records_asFloats()
        self.analyzer = FinancialAnalyzer(self.records) if self.records else None

    def add_expense(self, income: float, expenses: dict) -> dict:
        """Add a new expense entry."""
        record = FinanceData.add_record(income, expenses)
        self.records = FinanceData.get_records_asFloats()
        self.analyzer = FinancialAnalyzer(self.records)
        return record

    def get_dashboard_data(self) -> dict:
        """Get all data needed for the dashboard."""
        if not self.records:
            return {"status": "no_data"}

        summary = self.analyzer.calculate_summary()
        category_avgs = self.analyzer.get_category_averages()
        trend = self.analyzer.get_savings_trend()
        latest = self.records[-1]
        warnings = self.analyzer.detect_overspending(latest)
        recommender = BudgetRecommender(latest["income"])
        comparison = recommender.compare_to_actual(latest)

        return {
            "status": "ok",
            "summary": summary,
            "category_averages": category_avgs,
            "trend": trend,
            "latest_record": latest,
            "warnings": warnings,
            "budget_comparison": comparison,
            "recommendations": recommender.get_recommendations()
        }

    def predict_savings(self, features: np.ndarray) -> Optional[float]:
        """Predict savings using ML model (imported from model.py)."""
        try:
            from model import SavingsPredictor
            predictor = SavingsPredictor()
            return predictor.predict(features)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None


def get_expense_categories() -> list:
    """Return list of expense categories."""
    return ["food", "rent", "travel", "shopping", "utilities", "entertainment", "healthcare", "other"]


def cli_interface():
    """Interactive CLI for the finance coach."""
    FinanceData.initialize_csv()
    coach = FinanceCoach()
    categories = get_expense_categories()

    while True:
        print("\n" + "=" * 50)
        print("Personal Finance Coach")
        print("=" * 50)
        print("1. Add Monthly Expense")
        print("2. View Dashboard Summary")
        print("3. View All Records")
        print("4. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            print("\n--- Add Expense ---")
            income = float(input("Monthly income: $"))

            expenses = {}
            print("\nEnter expenses for each category:")
            for cat in categories:
                val = input(f"  {cat.capitalize()}: $")
                expenses[cat] = float(val) if val else 0.0

            record = coach.add_expense(income, expenses)
            print(f"\n✓ Record saved!")
            print(f"  Total Expense: ${record['total_expense']:.2f}")
            print(f"  Savings: ${record['savings']:.2f}")

            # Show warnings
            analyzer = FinancialAnalyzer([record])
            warnings = analyzer.detect_overspending(record)
            if warnings:
                print("\n⚠ Warnings:")
                for w in warnings:
                    print(f"  - {w}")

        elif choice == "2":
            data = coach.get_dashboard_data()
            if data["status"] == "no_data":
                print("No data available. Add expenses first.")
                continue

            summary = data["summary"]
            print("\n--- Financial Summary ---")
            print(f"Total Records: {summary['record_count']}")
            print(f"Average Monthly Income: ${summary['avg_monthly_income']:.2f}")
            print(f"Average Monthly Expenses: ${summary['avg_monthly_expenses']:.2f}")
            print(f"Average Monthly Savings: ${summary['avg_monthly_savings']:.2f}")
            print(f"Savings Rate: {summary['savings_rate']:.1f}%")

            print(f"\n--- Savings Trend ---")
            trend = data["trend"]
            print(f"Trend: {trend['trend'].upper()}")

            if trend["trend"] != "insufficient_data":
                print(f"Recent Avg: ${trend['recent_avg']:.2f}")
                print(f"Change: {trend['change_pct']:+.1f}%")

        elif choice == "3":
            records = FinanceData.get_records_asFloats()
            if not records:
                print("No records found.")
                continue

            print("\n--- All Records ---")
            for r in records:
                print(f"{r['date']} | Income: ${r['income']:.2f} | "
                      f"Expenses: ${r['total_expense']:.2f} | Savings: ${r['savings']:.2f}")

        elif choice == "4":
            print("\nThank you for using Personal Finance Coach!")
            break


if __name__ == "__main__":
    cli_interface()
