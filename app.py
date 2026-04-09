"""
Personal Finance Coach - Streamlit UI
Interactive web dashboard for the finance coach application.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from main import FinanceCoach, FinanceData, BudgetRecommender, get_expense_categories
from model import SavingsPredictor


def set_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="AI Personal Finance Coach",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {font-size: 2.5rem; font-weight: bold; color: #1E88E5;}
        .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
        .warning-text {color: #ff6b6b; font-weight: bold;}
        .success-text {color: #51cf66; font-weight: bold;}
        </style>
    """, unsafe_allow_html=True)


def plot_expense_pie(record: dict):
    """Create pie chart of expense distribution."""
    categories = get_expense_categories()
    values = [record.get(cat, 0) for cat in categories]

    # Filter out zero values
    non_zero = [(cat, val) for cat, val in zip(categories, values) if val > 0]
    if not non_zero:
        return None

    labels, sizes = zip(*non_zero)

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9', '#fd79a8', '#a29bfe']
    ax.pie(sizes, labels=[l.capitalize() for l in labels], autopct='%1.1f%%',
           colors=colors[:len(labels)], startangle=90)
    ax.set_title('Expense Distribution', fontsize=12, fontweight='bold')
    return fig


def plot_savings_trend(records: list):
    """Create line chart of savings over time."""
    if len(records) < 2:
        return None

    dates = [r['date'] for r in records]
    savings = [r['savings'] for r in records]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, savings, marker='o', linewidth=2, color='#1E88E5', markersize=6)
    ax.fill_between(dates, savings, alpha=0.3, color='#1E88E5')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Savings ($)')
    ax.set_title('Savings Trend Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_budget_comparison(record: dict, income: float):
    """Create bar chart comparing actual vs recommended budgets."""
    recommender = BudgetRecommender(income)
    rec = recommender.get_recommendations()

    categories = ['Needs\n(50%)', 'Wants\n(30%)', 'Savings\n(20%)']
    recommended = [rec['needs_budget'], rec['wants_budget'], rec['savings_target']]

    from main import FinancialAnalyzer
    analyzer = FinancialAnalyzer([record])
    categorized = analyzer.categorize_expenses(record)
    actual = [categorized['needs'], categorized['wants'], record['savings']]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, recommended, width, label='Recommended', color='#4ecdc4')
    bars2 = ax.bar(x + width/2, actual, width, label='Actual', color='#ff6b6b')

    ax.set_ylabel('Amount ($)')
    ax.set_title('Actual vs Recommended Budget (50/30/20 Rule)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'${height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'${height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    plt.tight_layout()
    return fig


def render_sidebar():
    """Render sidebar navigation and data entry form."""
    st.sidebar.title("💰 Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Add Expense", "History", "AI Predictions"])

    with st.sidebar.expander("➕ Quick Add Expense", expanded=True):
        income = st.number_input("Monthly Income ($)", min_value=0.0, step=100.0, key="quick_income")
        categories = get_expense_categories()
        expenses = {}
        for cat in categories:
            expenses[cat] = st.number_input(f"  {cat.capitalize()} ($)", min_value=0.0, step=10.0, key=f"quick_{cat}")

        if st.button("Save Entry", type="primary"):
            if income > 0:
                coach = FinanceCoach()
                coach.add_expense(income, expenses)
                st.success("✓ Expense entry saved!")
                st.rerun()
            else:
                st.error("Please enter a valid income")

    return page


def render_dashboard(coach: FinanceCoach):
    """Render the main dashboard page."""
    st.header("📊 Dashboard")

    data = coach.get_dashboard_data()

    if data["status"] == "no_data":
        st.info("👋 Welcome! Start by adding your first expense entry in the sidebar.")
        return

    # Key metrics row
    summary = data["summary"]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Income", f"${summary['total_income']:,.2f}",
                  help="Sum of all recorded income")
    with col2:
        st.metric("Total Expenses", f"${summary['total_expenses']:,.2f}",
                  delta=f"-{summary['avg_monthly_expenses']:.0f}/mo avg")
    with col3:
        st.metric("Total Savings", f"${summary['total_savings']:,.2f}",
                  delta=f"{summary['savings_rate']:.1f}% rate")
    with col4:
        trend = data["trend"]
        trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️", "insufficient_data": "⏳"}
        st.metric("Savings Trend", trend_emoji.get(trend["trend"], "❓"),
                  delta=f"{trend['change_pct']:+.1f}%" if trend["trend"] != "insufficient_data" else None)

    st.divider()

    # Warnings
    if data["warnings"]:
        st.subheader("⚠️ Financial Alerts")
        for warning in data["warnings"]:
            st.warning(warning)

    # Charts row
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("💸 Expense Distribution")
        fig = plot_expense_pie(data["latest_record"])
        if fig:
            st.pyplot(fig)
        else:
            st.info("No expense data to display")

    with col_right:
        st.subheader("📈 Budget Comparison")
        fig = plot_budget_comparison(data["latest_record"], data["latest_record"]["income"])
        if fig:
            st.pyplot(fig)

    # Savings trend
    records = FinanceData.get_records_asFloats()
    if len(records) >= 2:
        st.subheader("📉 Savings Over Time")
        fig = plot_savings_trend(records)
        if fig:
            st.pyplot(fig)

    # Budget recommendations
    st.divider()
    st.subheader("🎯 Budget Recommendations (50/30/20 Rule)")
    rec = data["recommendations"]
    comparison = data["budget_comparison"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Needs (50%)", f"${rec['needs_budget']:.2f}",
                  delta=f"{comparison['needs_diff']:+.2f}" if comparison['needs_diff'] != 0 else None,
                  delta_color="inverse")
    with col2:
        st.metric("Wants (30%)", f"${rec['wants_budget']:.2f}",
                  delta=f"{comparison['wants_diff']:+.2f}" if comparison['wants_diff'] != 0 else None,
                  delta_color="inverse")
    with col3:
        st.metric("Savings (20%)", f"${rec['savings_target']:.2f}",
                  delta=f"{comparison['savings_diff']:+.2f}" if comparison['savings_diff'] != 0 else None,
                  delta_color="normal")


def render_add_expense():
    """Render the add expense page."""
    st.header("➕ Add New Expense Entry")

    with st.form("expense_form"):
        st.subheader("Income")
        income = st.number_input("Monthly Income ($)", min_value=0.0, step=100.0)

        st.subheader("Expenses by Category")
        categories = get_expense_categories()

        col1, col2 = st.columns(2)
        expenses = {}

        for i, cat in enumerate(categories):
            with col1 if i % 2 == 0 else col2:
                expenses[cat] = st.number_input(f"  {cat.capitalize()} ($)", min_value=0.0, step=10.0)

        submitted = st.form_submit_button("💾 Save Entry", type="primary", use_container_width=True)

        if submitted:
            if income <= 0:
                st.error("Please enter a valid income")
            else:
                coach = FinanceCoach()
                record = coach.add_expense(income, expenses)

                st.success("✓ Expense entry saved successfully!")
                st.json({
                    "Total Expense": f"${record['total_expense']:.2f}",
                    "Savings": f"${record['savings']:.2f}",
                    "Savings Rate": f"{(record['savings']/income*100):.1f}%"
                })

                # Show warnings
                from main import FinancialAnalyzer
                analyzer = FinancialAnalyzer([record])
                warnings = analyzer.detect_overspending(record)
                if warnings:
                    st.warning("⚠️ " + " ".join(warnings))


def render_history():
    """Render the history page with all records."""
    st.header("📋 Expense History")

    records = FinanceData.get_records_asFloats()

    if not records:
        st.info("No records found. Add your first expense entry.")
        return

    # Convert to DataFrame for display
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=False)

    # Display summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Entries", len(records))
    with col2:
        avg_savings = df["savings"].mean()
        st.metric("Average Savings", f"${avg_savings:.2f}")
    with col3:
        best = df.loc[df["savings"].idxmax()] if len(df) > 0 else None
        if best is not None:
            st.metric("Best Savings", f"${best['savings']:.2f}", delta=best['date'])

    st.divider()

    # Display table
    st.dataframe(
        df.style.format({
            "income": "${:,.2f}",
            "food": "${:,.2f}",
            "rent": "${:,.2f}",
            "travel": "${:,.2f}",
            "shopping": "${:,.2f}",
            "utilities": "${:,.2f}",
            "entertainment": "${:,.2f}",
            "healthcare": "${:,.2f}",
            "other": "${:,.2f}",
            "savings": "${:,.2f}",
            "total_expense": "${:,.2f}"
        }, na_rep="-"),
        use_container_width=True,
        hide_index=True
    )

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download CSV",
        csv,
        file_name=f"finance_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


def render_predictions():
    """Render the AI predictions page."""
    st.header("🤖 AI Savings Predictions")

    try:
        predictor = SavingsPredictor()
    except Exception as e:
        st.error(f"Could not initialize predictor: {e}")
        return

    records = FinanceData.get_records_asFloats()

    if len(records) < 3:
        st.info("👋 Need at least 3 months of data for predictions. Keep adding expenses!")
        return

    # Model evaluation
    st.subheader("📊 Model Performance")
    eval_data = predictor.evaluate()

    if "error" not in eval_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute Error", f"${eval_data['mae']:.2f}")
        with col2:
            st.metric("R² Score", f"{eval_data['r2']:.3f}")
        with col3:
            st.metric("Training Samples", eval_data["sample_count"])

    # Feature importance
    st.subheader("🔍 What Affects Your Savings Most?")
    importance = predictor.get_feature_importance()

    if importance:
        # Sort by absolute value
        sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        labels = [f"{k.capitalize()}: {v:+.2f}" for k, v in sorted_imp]
        values = [abs(v) for v in importance.values()]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#1E88E5' if v >= 0 else '#ff6b6b' for v in importance.values()]
        ax.barh(list(importance.keys()), list(importance.values()), color=colors)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Coefficient (Impact on Savings)')
        ax.set_title('Feature Importance in Savings Prediction', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        st.caption("Positive = increases savings | Negative = decreases savings")

    # Manual prediction
    st.divider()
    st.subheader("🔮 Try a Prediction")

    with st.form("prediction_form"):
        st.write("Enter hypothetical expenses to predict savings:")
        income = st.number_input("Income ($)", min_value=0.0, step=100.0, key="pred_income")
        categories = get_expense_categories()

        col1, col2 = st.columns(2)
        expenses_pred = {}
        for i, cat in enumerate(categories):
            with col1 if i % 2 == 0 else col2:
                expenses_pred[cat] = st.number_input(f"  {cat.capitalize()} ($)", min_value=0.0, step=10.0, key=f"pred_{cat}")

        submitted = st.form_submit_button("🔮 Predict Savings", use_container_width=True)

        if submitted:
            if income > 0:
                prediction = predictor.predict_from_expenses(income, expenses_pred)
                actual_savings = income - sum(expenses_pred.values())

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Savings (ML)", f"${prediction:.2f}")
                with col2:
                    st.metric("Simple Calculation", f"${actual_savings:.2f}")

                if prediction != actual_savings:
                    diff = prediction - actual_savings
                    st.info(f"The ML model predicts {'higher' if diff > 0 else 'lower'} savings than simple math, "
                            f"based on learned patterns in your data.")


def main():
    """Main Streamlit application."""
    set_page_config()

    st.title("💰 AI Personal Finance Coach")
    st.caption("Track, analyze, and predict your savings with AI")

    page = render_sidebar()

    coach = FinanceCoach()

    if page == "Dashboard":
        render_dashboard(coach)
    elif page == "Add Expense":
        render_add_expense()
    elif page == "History":
        render_history()
    elif page == "AI Predictions":
        render_predictions()


if __name__ == "__main__":
    main()
