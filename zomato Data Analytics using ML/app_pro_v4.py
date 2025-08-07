import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from textblob import TextBlob
from wordcloud import WordCloud

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="FOODPRINTS ", layout="wide")

# 📤 Upload CSV
st.sidebar.title("📤 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your Zomato CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Uploaded successfully!")
else:
    df = pd.read_csv("data/zomato.csv", encoding="latin1")

# Clean & preprocess
df.dropna(subset=['restaurant name'], inplace=True)
df['avg cost (two people)'] = pd.to_numeric(df['avg cost (two people)'], errors='coerce')
df['num of ratings'] = pd.to_numeric(df['num of ratings'], errors='coerce')
df['rate (out of 5)'] = pd.to_numeric(df['rate (out of 5)'], errors='coerce')

# 🔎 Sidebar Filters
st.sidebar.header("Filters")
areas = df['area'].dropna().unique().tolist()
selected_area = st.sidebar.selectbox("Select Area", ['All'] + sorted(areas))

cuisines = df['cuisines type'].dropna().str.split(', ').explode().unique().tolist()
selected_cuisine = st.sidebar.selectbox("Select Cuisine", ['All'] + sorted(cuisines))

price_range = st.sidebar.slider("Select Avg. Cost Range", 0, int(df['avg cost (two people)'].max()), (0, 1000))
rating_range = st.sidebar.slider("Select Rating Range", 0.0, 5.0, (3.0, 5.0), step=0.1)

if 'restaurant type' in df.columns:
    rest_types = df['restaurant type'].dropna().unique().tolist()
    selected_type = st.sidebar.selectbox("Select Restaurant Type", ['All'] + sorted(rest_types))
else:
    selected_type = 'All'

cost_categories = ["All", "Low (<300)", "Medium (300-700)", "High (>700)"]
selected_cost = st.sidebar.selectbox("Select Cost Category", cost_categories)

#  Apply Filters
filtered_df = df.copy()
if selected_area != 'All':
    filtered_df = filtered_df[filtered_df['area'] == selected_area]
if selected_cuisine != 'All':
    filtered_df = filtered_df[filtered_df['cuisines type'].str.contains(selected_cuisine, na=False)]
if selected_type != 'All':
    filtered_df = filtered_df[filtered_df['restaurant type'] == selected_type]
filtered_df = filtered_df[
    (filtered_df['avg cost (two people)'] >= price_range[0]) &
    (filtered_df['avg cost (two people)'] <= price_range[1]) &
    (filtered_df['rate (out of 5)'] >= rating_range[0]) &
    (filtered_df['rate (out of 5)'] <= rating_range[1])
]
if selected_cost == "Low (<300)":
    filtered_df = filtered_df[filtered_df['avg cost (two people)'] < 300]
elif selected_cost == "Medium (300-700)":
    filtered_df = filtered_df[(filtered_df['avg cost (two people)'] >= 300) & (filtered_df['avg cost (two people)'] <= 700)]
elif selected_cost == "High (>700)":
    filtered_df = filtered_df[filtered_df['avg cost (two people)'] > 700]

#  Tabs
st.title("FOODPRINTS")
tab1, tab2, tab3, tab4 ,tab5 = st.tabs(["📊 Dashboard", "🗺️ Map", "🧠 Sentiment", "📥 Export","📈 Predict"])

#  Dashboard Tab
with tab1:
    st.title("📊 Restaurant Performance Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Restaurants", filtered_df['restaurant name'].nunique())
    col2.metric("Avg Rating", f"{filtered_df['rate (out of 5)'].mean():.2f} ⭐")
    col3.metric("Total Orders", int(filtered_df['num of ratings'].sum()))

    st.markdown("### 🏆 Top 10 Ordered Restaurants")
    top_rest = filtered_df.groupby('restaurant name')['num of ratings'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_rest)

    st.markdown("### 💰 Top 10 Estimated Revenue")
    rev_df = filtered_df.dropna(subset=['avg cost (two people)', 'num of ratings'])
    rev_df['estimated_revenue'] = rev_df['num of ratings'] * (rev_df['avg cost (two people)'] / 2)
    top_rev = rev_df.groupby('restaurant name')['estimated_revenue'].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_rev)

    st.markdown("### 📈 Cost vs Rating Correlation")
    scatter_df = filtered_df.dropna(subset=['avg cost (two people)', 'rate (out of 5)'])
    fig, ax = plt.subplots()
    ax.scatter(scatter_df['avg cost (two people)'], scatter_df['rate (out of 5)'], alpha=0.5, color='purple')
    ax.set_xlabel("Avg Cost for Two")
    ax.set_ylabel("Rating")
    ax.set_title("Cost vs Rating")
    st.pyplot(fig)

    # Add this block below the Cost vs Rating chart
    st.markdown("### 🛍️ Order Type Distribution")
    if 'online_order' in filtered_df.columns:
        order_counts = filtered_df['online_order'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(order_counts, labels=order_counts.index, autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')
        st.pyplot(fig1)

    st.markdown("### 🍜 Most Ordered Cuisines")
    if 'cuisines type' in filtered_df.columns:
        cuisine_series = filtered_df['cuisines type'].dropna().str.split(', ').explode()
        top_cuisines = cuisine_series.value_counts().head(10)
        st.bar_chart(top_cuisines)




# 🗺️ Map Tab
with tab2:
    st.title("🗺️ Restaurant Map View")
    if {'latitude', 'longitude'}.issubset(filtered_df.columns):
        m = folium.Map(location=[filtered_df['latitude'].mean(), filtered_df['longitude'].mean()], zoom_start=12)
        cluster = MarkerCluster().add_to(m)
        for _, row in filtered_df.iterrows():
            if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"{row['restaurant name']} ({row['rate (out of 5)']}⭐)",
                    icon=folium.Icon(color="red", icon="cutlery", prefix="fa")
                ).add_to(cluster)
        st_folium(m, width=700, height=500)
    else:
        st.warning("This dataset has no latitude/longitude columns.")

# 🧠 Sentiment Tab
with tab3:
    st.title("🧠 Review Sentiment Classifier")
    if 'reviews' in filtered_df.columns:
        reviews = filtered_df['reviews'].dropna().astype(str)
        sentiments = reviews.apply(lambda x: TextBlob(x).sentiment.polarity)
        labels = sentiments.apply(lambda p: 'Positive' if p > 0.1 else ('Negative' if p < -0.1 else 'Neutral'))
        sentiment_df = pd.DataFrame({'Review': reviews, 'Sentiment': labels})
        st.bar_chart(sentiment_df['Sentiment'].value_counts())

        st.subheader("☁️ Word Cloud from Reviews")
        text = " ".join(reviews)
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.warning("No reviews found in dataset.")

# 📥 Export Tab
with tab4:
    st.title("📥 Download Filtered Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download PDF", data=csv, file_name="filtered_zomato.pdf", mime='text/pdf')

    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download WORD", data=csv, file_name="filtered_zomato.word", mime='text/word')

   
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download EXCEL", data=csv, file_name="filtered_zomato.xlxs", mime='text/excel')

    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="filtered_zomato.csv", mime='text/csv')

# 📈 ML Rating Prediction Tab
with tab5:
    st.title("📈 Predict Restaurant Rating")
    st.markdown("This model predicts a restaurant's rating based on cost, area, cuisine, and restaurant type.")

    ml_df = filtered_df.dropna(subset=[
        'rate (out of 5)', 'avg cost (two people)', 'area',
        'restaurant type', 'cuisines type'
    ])

    if ml_df.shape[0] > 100:
        # Features and target
        X = ml_df[['avg cost (two people)', 'area', 'restaurant type', 'cuisines type']]
        y = ml_df['rate (out of 5)']

        # Preprocessing
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['area', 'restaurant type', 'cuisines type'])
        ], remainder='passthrough')

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        model.fit(X, y)

        # Input Form
        with st.form("predict_form"):
            cost_input = st.number_input("Avg Cost for Two", min_value=50, max_value=2000, value=300)
            area_input = st.selectbox("Area", df['area'].dropna().unique())
            type_input = st.selectbox("Restaurant Type", df['restaurant type'].dropna().unique())
            cuisine_input = st.selectbox("Cuisine Type", df['cuisines type'].dropna().str.split(', ').explode().unique())
            submitted = st.form_submit_button("Predict Rating")

            if submitted:
                pred_input = pd.DataFrame({
                    'avg cost (two people)': [cost_input],
                    'area': [area_input],
                    'restaurant type': [type_input],
                    'cuisines type': [cuisine_input]
                })
                prediction = model.predict(pred_input)[0]
                st.success(f"⭐ Predicted Rating: {round(prediction, 2)} / 5")
    else:
        st.warning("Not enough data to train the model. Please adjust your filters or upload more data.")


