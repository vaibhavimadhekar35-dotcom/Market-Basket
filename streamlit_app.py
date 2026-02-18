import streamlit as st
import pandas as pd

st.title("🛍️ Product Recommendation System")
st.write("Select a product to get recommendations")

# ===============================
# Load Rules
# ===============================
rules = pd.read_csv("association_rules.csv")

# Clean frozenset text manually
rules['antecedents'] = rules['antecedents'].str.replace(
    "frozenset({", "", regex=False
).str.replace("})", "", regex=False)

rules['consequents'] = rules['consequents'].str.replace(
    "frozenset({", "", regex=False
).str.replace("})", "", regex=False)

# Remove quotes
rules['antecedents'] = rules['antecedents'].str.replace("'", "")
rules['consequents'] = rules['consequents'].str.replace("'", "")

# Get unique products
products = sorted(rules['antecedents'].unique())

selected_product = st.selectbox("Choose Product:", products)

# ===============================
# Recommendation Logic
# ===============================
if selected_product:

    recommendations = rules[
    rules['antecedents'].str.contains(selected_product, regex=False)
     ].sort_values(by="confidence", ascending=False)


    if not recommendations.empty:
        st.subheader("Recommended Products:")

        for _, row in recommendations.head(5).iterrows():
            st.write(
                f"👉 {row['consequents']} "
                f"(Confidence: {round(row['confidence'],2)}, "
                f"Lift: {round(row['lift'],2)})"
            )
    else:
        st.warning("No recommendations found.")
