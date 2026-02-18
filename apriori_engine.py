import pandas as pd

# ===============================
# 1️⃣ Load Dataset
# ===============================
df = pd.read_csv("online.csv", encoding='utf-8-sig')

# ===============================
# 🔎 Basic Exploration (EDA)
# ===============================
print("First 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# ===============================
# 2️⃣ Basic Cleaning
# ===============================

# Remove duplicates
df = df.drop_duplicates()

# Remove cancelled invoices
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove invalid values
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Remove rows without CustomerID
df = df.dropna(subset=['CustomerID'])

# Fix datatypes
df['CustomerID'] = df['CustomerID'].astype(int)

# Convert InvoiceDate properly
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Drop rows where date failed
df = df.dropna(subset=['InvoiceDate'])

# Fill missing descriptions
df['Description'] = df['Description'].fillna("Unknown")

# ===============================
# 🔎 Re-check After Cleaning
# ===============================
print("\nShape After Cleaning:", df.shape)
print("\nDataset Info After Cleaning:")
print(df.info())
print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# ===============================
# 3️⃣ Feature Engineering
# ===============================

df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day

# Convert to category (memory optimization)
df['Country'] = df['Country'].astype('category')
df['StockCode'] = df['StockCode'].astype('category')

print("\nFinal Dataset Shape:", df.shape)

# ===============================
# 4️⃣ Create Basket Matrix
# ===============================

basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'] \
           .sum().unstack().fillna(0)

basket = (basket > 0)

print("Basket Shape:", basket.shape)

# ===============================
# 5️⃣ Apply Apriori
# ===============================

from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(
    basket,
    min_support=0.02,
    use_colnames=True
)

print("\nFrequent Itemsets Found:", len(frequent_itemsets))

# ===============================
# 6️⃣ Generate Association Rules
# ===============================

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.03  
)

rules = rules[(rules['confidence'] > 0.05) & (rules['lift'] > 1)]


print("\nTop Association Rules:\n")
print(rules[['antecedents', 'consequents',
             'support', 'confidence', 'lift']].head(10))

# ===============================
# 7️⃣ Save Results
# ===============================

rules.to_csv("association_rules.csv", index=False)

print("\nAssociation rules saved successfully!")
