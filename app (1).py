import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import requests
from io import StringIO

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set page configuration for a better look
st.set_page_config(page_title="E-Commerce RFM Dashboard", layout="wide")

# Title and description
st.title("📊 E-Commerce RFM Analysis Dashboard")
st.write("""
Welcome to the interactive dashboard for analyzing e-commerce sales data. 
This dataset, sourced from Kaggle[](https://www.kaggle.com/datasets/carrie1/ecommerce-data), 
is used for RFM (Recency, Frequency, Monetary) analysis to segment customers and derive business insights.
""")

# Cache data loading for performance
@st.cache_data
def load_data():
    try:
        file_id = '1yKSLfyic5lzeUFWk1cxR9fqrYAUsnlUo'  # ID dari link Anda
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        # Gunakan requests untuk fetch konten (hindari parser error jika ada warning Google)
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Error fetching file: Status {response.status_code}. Pastikan file publik!")
            return pd.DataFrame()
        # Cek jika respons adalah HTML (error umum)
        if '<html' in response.text.lower() or 'drive.google.com' in response.text.lower():
            st.error("File mengembalikan HTML, bukan CSV. Set file ke 'Anyone with the link' di Google Drive.")
            return pd.DataFrame()
        # Parse sebagai CSV
        df = pd.read_csv(StringIO(response.text), encoding='latin1', encoding_errors='ignore', low_memory=False)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        # Filter positive quantity and price, remove NaN CustomerID
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['CustomerID'].notna())]
        st.success(f"Data loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}. Cek sharing settings di Google Drive.")
        return pd.DataFrame()  # Return empty DF to prevent crash

# Load dataset
df = load_data()

if df.empty:
    st.stop()  # Stop if no data

# Sidebar for interactivity
st.sidebar.header("Filter Options")
country_filter = st.sidebar.selectbox("Select Country", options=["All"] + sorted(list(df['Country'].unique())))

# Filter data based on selection
if country_filter != "All":
    filtered_df = df[df['Country'] == country_filter]
else:
    filtered_df = df

# Section: Dataset Exploration
st.header("🔍 Dataset Exploration")
st.write("Below is a random sample of 10 records from the dataset:")
st.dataframe(filtered_df.sample(10, random_state=123))

# RFM Calculation
st.header("📈 RFM Analysis and Clustering")

# Calculate RFM - Use fixed 'today' for consistency (end of dataset period)
today = dt.date(2012, 1, 1)  # Dataset ends Dec 2011
rfm = filtered_df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (today - x.max()).days,
    'InvoiceNo': 'nunique',  # Frequency as unique invoices
    'TotalPrice': 'sum'      # Monetary
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

# Remove any negative or zero monetary
rfm = rfm[rfm['Monetary'] > 0]

if not rfm.empty:
    # Scale RFM
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Section: Elbow Method
    st.subheader("Metode Elbow untuk Menentukan Klaster Optimal")
    sse = []
    max_k = min(5, len(rfm) - 1)  # Limit to 5 for faster computation
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        sse.append(kmeans.inertia_)

    fig_elbow, ax = plt.subplots()
    ax.plot(range(1, max_k + 1), sse, marker='o', color='blue', linestyle='--')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('SSE')
    ax.set_title('Elbow Method for Optimal Clusters')
    ax.grid(True)
    st.pyplot(fig_elbow)

    # Assume optimal clusters = 4 (common for RFM)
    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Label clusters based on avg RFM (improved: lower recency + higher freq/mon = better segment)
    avg_rfm = rfm.groupby('Cluster').mean(numeric_only=True).reset_index()
    # Custom score: low Recency (good), high Freq/Mon (good)
    avg_rfm['Score'] = -avg_rfm['Recency'] + avg_rfm['Frequency'] + (avg_rfm['Monetary'] / 1000)  # Normalize monetary scale
    avg_rfm = avg_rfm.sort_values('Score', ascending=False)
    labels = ['Champions', 'Loyal Customers', 'At Risk', 'New Customers']  # Adjusted order
    label_map = {avg_rfm.iloc[i]['Cluster']: labels[i] for i in range(len(labels))}
    rfm['Segment'] = rfm['Cluster'].map(label_map)
    avg_rfm['Segment'] = avg_rfm['Cluster'].map(label_map)

    # Visual: Average RFM per Cluster
    st.subheader("Rata-rata Nilai RFM per Klaster")
    avg_rfm_melt = pd.melt(avg_rfm, id_vars=['Cluster', 'Segment'], value_vars=['Recency', 'Frequency', 'Monetary'])
    fig_rfm = px.bar(avg_rfm_melt, x='Segment', y='value', color='variable', barmode='group',
                     title='Rata-rata Nilai RFM per Klaster',
                     color_discrete_map={'Recency': 'red', 'Frequency': 'green', 'Monetary': 'blue'})
    st.plotly_chart(fig_rfm, use_container_width=True)

    # Visual: Distribusi Pelanggan berdasarkan Segmen RFM
    st.subheader("Distribusi Pelanggan berdasarkan Segmen RFM")
    customer_dist = rfm['Segment'].value_counts().reset_index()
    customer_dist.columns = ['Segment', 'Count']
    fig_dist = px.pie(customer_dist, values='Count', names='Segment', title='Distribusi Pelanggan per Segmen',
                      color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Visual: Total Kontribusi Revenue per Segmen Pelanggan
    st.subheader("Total Kontribusi Revenue per Segmen Pelanggan")
    revenue_contrib = rfm.groupby('Segment')['Monetary'].sum().reset_index()
    fig_revenue = px.bar(revenue_contrib, x='Segment', y='Monetary', title='Kontribusi Revenue per Segmen',
                         color='Segment', color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig_revenue, use_container_width=True)

# Revenue per Month
st.header("📅 Revenue Analysis")
st.subheader("Revenue per Month")
filtered_df['Month'] = filtered_df['InvoiceDate'].dt.to_period('M').astype(str)
monthly_revenue = filtered_df.groupby('Month')['TotalPrice'].sum().reset_index()
fig_monthly = px.line(monthly_revenue, x='Month', y='TotalPrice', title='Revenue per Bulan',
                      line_shape='spline', markers=True, color_discrete_sequence=['purple'])
st.plotly_chart(fig_monthly, use_container_width=True)

# Section: Geographic Sales Visualization
st.header("🌍 Geographic Sales Distribution")
st.subheader("Distribusi Revenue berdasarkan Negara (Choropleth)")
df_country = filtered_df.groupby('Country')['TotalPrice'].sum().reset_index()
fig_geo = px.choropleth(
    df_country,
    locations='Country',
    locationmode='country names',
    color='TotalPrice',
    hover_name='Country',
    title='Total Revenue per Country',
    color_continuous_scale=px.colors.sequential.Viridis
)
st.plotly_chart(fig_geo, use_container_width=True)

st.subheader("Distribusi Revenue berdasarkan Negara (Bar Chart)")
fig_country_bar = px.bar(df_country.sort_values('TotalPrice', ascending=False), 
                         x='Country', y='TotalPrice', title='Revenue per Negara',
                         color='TotalPrice', color_continuous_scale=px.colors.sequential.Inferno)
st.plotly_chart(fig_country_bar, use_container_width=True)

# Section: Conclusions and Recommendations
st.header("📝 Conclusions and Recommendations")
st.markdown("""
After conducting data understanding, cleaning, feature engineering, customer segmentation, and analysis, 
the main issue in this e-commerce business is treating all customers equally, which is inefficient and risks losing valuable customers.

**Key Findings:**
1. A small group of products (e.g., paper craft sets, cake stands) and 'Champions' customers drive the majority of revenue. These are the business's core assets and must be maintained.
2. The business excels at acquiring new customers, but many only purchase once. Additionally, valuable 'At Risk' customers are starting to shop less frequently, which needs immediate attention.
3. Customers show clear patterns, e.g., those buying green teacups are likely to want pink and red ones, indicating they are collecting sets.
4. Most sales come from the UK, which is strong but risky if the market weakens. Positive signals from countries like Germany and the Netherlands suggest growth opportunities.

**Recommended Actions:**
1. **Differentiate Customer Treatment**: Offer rewards to 'Champions' and targeted discounts to 'At Risk' customers to re-engage them.
2. **Retain Top Customers**: Prioritize budget and effort to keep 'Champions' loyal and re-engage 'At Risk' customers, as retaining existing customers is cheaper than acquiring new ones.
3. **Smarter Product Sales**: Create bundling packages for collectible item sets to encourage repeat purchases.
4. **Expand Internationally**: Target markets outside the UK (e.g., Germany, Netherlands) with promotions like free shipping to diversify revenue sources.
""")

# Footer
st.write("---")
st.write("Built with Streamlit | Data Source: Kaggle E-Commerce Dataset")