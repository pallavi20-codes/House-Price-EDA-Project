
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="House Price EDA Dashboard",
    page_icon="🏠",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()

st.title("🏠 House Price Exploratory Data Analysis")
st.write("An interactive dashboard to explore housing data and discover price patterns.")

st.sidebar.header("Dashboard Filters")

quality = st.sidebar.multiselect(
    "Select Overall Quality",
    sorted(df["OverallQual"].dropna().unique()),
    default=sorted(df["OverallQual"].dropna().unique())
)

filtered = df[df["OverallQual"].isin(quality)]

st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Houses", f"{len(filtered):,}")
c2.metric("Average Price", f"${filtered['SalePrice'].mean():,.0f}")
c3.metric("Median Price", f"${filtered['SalePrice'].median():,.0f}")
c4.metric("Average Living Area", f"{filtered['GrLivArea'].mean():,.0f} sq ft")

st.divider()

st.subheader("House Price Distribution")

fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(filtered["SalePrice"], kde=True, ax=ax)
ax.set_xlabel("Sale Price ($)")
ax.set_title("Distribution of House Prices")
st.pyplot(fig)
plt.close(fig)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Living Area vs Sale Price")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(
        data=filtered,
        x="GrLivArea",
        y="SalePrice",
        ax=ax
    )
    ax.set_title("Living Area vs Sale Price")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Quality vs Average Price")
    quality_price = filtered.groupby("OverallQual")["SalePrice"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    quality_price.plot(kind="bar", ax=ax)
    ax.set_xlabel("Overall Quality")
    ax.set_ylabel("Average Sale Price ($)")
    st.pyplot(fig)
    plt.close(fig)

st.divider()

st.subheader("Correlation Heatmap")

numeric = filtered.select_dtypes(include=np.number)
selected = [
    col for col in [
        "SalePrice", "OverallQual", "GrLivArea",
        "GarageCars", "TotalBsmtSF", "YearBuilt",
        "FullBath", "LotArea"
    ] if col in numeric.columns
]

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    numeric[selected].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)
st.pyplot(fig)
plt.close(fig)

st.divider()

st.subheader("Dataset Information")
st.write("Number of rows:", filtered.shape[0])
st.write("Number of columns:", filtered.shape[1])
st.write("Missing values:", int(filtered.isnull().sum().sum()))

with st.expander("View Dataset"):
    st.dataframe(filtered.head(100))

st.download_button(
    "Download Filtered Dataset",
    data=filtered.to_csv(index=False),
    file_name="house_price_data.csv",
    mime="text/csv"
)

st.success("House Price EDA Dashboard loaded successfully!")
