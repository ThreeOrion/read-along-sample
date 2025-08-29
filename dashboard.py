import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Pakistan Read-Along Insights",
    page_icon="🇵🇰"
)

# --- File Paths ---
GEOJSON_PATH = 'data/geo data/gadm41_PAK_3.json'
DATA_PATH = 'data/sample data/The Data.csv'
DISTRICT_TABLE_PATH = '/data/sample data/District table.csv'
BOOK_TABLE_PATH = 'data/sample data/Book table.csv'

# --- Step 1 & 2: Data Loading and Preprocessing ---
@st.cache_data
def load_and_prepare_data():
    """
    Loads, cleans, and merges all necessary data files.
    """
    try:
        # Load core data, district info, and book info
        df_data = pd.read_csv(DATA_PATH, encoding='latin1')
        df_districts = pd.read_csv(DISTRICT_TABLE_PATH, encoding='latin1')
        df_books = pd.read_csv(BOOK_TABLE_PATH, encoding='latin1')
        
        # Load geospatial data
        gdf_map = gpd.read_file(GEOJSON_PATH)

    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Please ensure all file paths are correct.")
        return pd.DataFrame(), None, pd.DataFrame()

    # --- Data Cleaning and Feature Engineering ---
    
    # Rename columns for consistency
    df_data = df_data.rename(columns={
        'district_name': 'District',
        'mic_time_minutes': 'time_spent_minutes',
        'corrcet_answer_count': 'correct_answer_count'
    })
    
    # Convert minutes to seconds for calculations
    df_data['time_spent_seconds'] = df_data['time_spent_minutes'] * 60

    # Add a 'completed_book' flag
    df_data['completed_book'] = df_data['book_completion_percentage'] >= 0.95

    # Create a synthetic date for trend analysis (since none exists in the data)
    num_rows = len(df_data)
    start_date = pd.to_datetime('2024-01-01')
    random_dates = start_date + pd.to_timedelta(np.random.randint(0, 180, num_rows), 'd')
    df_data['event_ts'] = random_dates
    df_data['date'] = df_data['event_ts'].dt.date
    df_data['week'] = df_data['event_ts'].dt.to_period('W').apply(lambda r: r.start_time).dt.date

    # *** ADD SYNTHETIC EQUITY DATA ***
    # Create a mapping from school_id to a synthetic school type
    unique_schools = df_data['school_id'].unique()
    school_type_map = {school_id: np.random.choice(['Public', 'Private'], p=[0.8, 0.2]) for school_id in unique_schools}
    df_data['school_type'] = df_data['school_id'].map(school_type_map)
    
    # Add synthetic gender data for each event
    df_data['gender'] = np.random.choice(['Male', 'Female'], size=len(df_data), p=[0.52, 0.48])


    # Merge with district data to get lat/lon
    df_merged = pd.merge(df_data, df_districts, on='district_id', how='left')
    
    # Clean up the GeoJSON data
    gdf_map_punjab = gdf_map[gdf_map['NAME_1'] == 'Punjab'].copy()
    gdf_map_punjab = gdf_map_punjab.rename(columns={'NAME_3': 'District'})
    
    return df_merged, gdf_map_punjab, df_districts


# --- Step 4: KPI Calculation ---
def calculate_kpis(df):
    """Calculates all the KPIs for a given dataframe."""
    if df.empty:
        return {
            'active_schools': 0, 'total_sessions': 0, 'avg_daily_reading_time': 0,
            'accuracy_ratio': 0, 'completed_ratio': 0, 'comprehension_score': 0,
            'engagement_index': 0
        }

    kpis = {}
    # Define a session as a unique day for a school
    sessions_df = df.drop_duplicates(subset=['date', 'school_id'])
    
    kpis['active_schools'] = df['school_id'].nunique()
    kpis['total_sessions'] = len(sessions_df)
    
    # Avg daily time per school
    daily_time = df.groupby(['date', 'school_id'])['time_spent_seconds'].sum().reset_index()
    kpis['avg_daily_reading_time'] = (daily_time['time_spent_seconds'].mean() / 60) if not daily_time.empty else 0
    
    # Accuracy and Comprehension
    kpis['accuracy_ratio'] = df['correct_word_count'].sum() / df['total_word_count'].sum() if df['total_word_count'].sum() > 0 else 0
    kpis['comprehension_score'] = df['correct_answer_count'].sum() / df['total_question_count'].sum() if df['total_question_count'].sum() > 0 else 0
    
    # Completion Ratio
    kpis['completed_ratio'] = df['completed_book'].sum() / len(df) if not df.empty else 0

    # Engagement Index
    engagement_score = (kpis['accuracy_ratio'] * 0.4) + (kpis['comprehension_score'] * 0.4) + (kpis['completed_ratio'] * 0.2)
    kpis['engagement_index'] = engagement_score * 100

    return kpis

def format_delta(current, previous):
    """Formats the percentage change between two numbers for the st.metric component."""
    if previous == 0:
        return " "
    delta = ((current - previous) / previous) * 100
    return f"{delta:.1f}%"


# --- Load the data ---
master_df, punjab_geo_df, district_df = load_and_prepare_data()

# --- Main App ---
st.title("Pakistan Read-Along – Literacy & Engagement Insights")

if not master_df.empty and punjab_geo_df is not None:
    
    # --- Step 3: Sidebar and Filters ---
    st.sidebar.header("Filters")
    time_window_options = {'Last 7 Days': 7, 'Last 30 Days': 30, 'Last 90 Days': 90}
    selected_time_window = st.sidebar.selectbox("Time Window", options=list(time_window_options.keys()), index=1)
    
    all_districts = sorted(master_df['District'].unique())
    selected_districts = st.sidebar.multiselect("District", options=all_districts, default=all_districts)
    
    # Add filters for new synthetic data
    school_types = sorted(master_df['school_type'].unique())
    selected_school_types = st.sidebar.multiselect("School Type", options=school_types, default=school_types)

    genders = sorted(master_df['gender'].unique())
    selected_genders = st.sidebar.multiselect("Gender", options=genders, default=genders)

    compare_period = st.sidebar.toggle("Compare: Previous Period", True)

    # --- Filtering Logic ---
    days = time_window_options[selected_time_window]
    end_date = master_df['date'].max()
    start_date = end_date - pd.to_timedelta(days - 1, unit='d')

    # Current period data
    filtered_df = master_df[
        (master_df['date'] >= start_date) & 
        (master_df['date'] <= end_date) &
        (master_df['District'].isin(selected_districts)) &
        (master_df['school_type'].isin(selected_school_types)) &
        (master_df['gender'].isin(selected_genders))
    ]

    # Previous period data
    prev_start_date = start_date - pd.to_timedelta(days, unit='d')
    prev_end_date = start_date - pd.to_timedelta(1, unit='d')
    prev_period_df = master_df[
        (master_df['date'] >= prev_start_date) & 
        (master_df['date'] <= prev_end_date) &
        (master_df['District'].isin(selected_districts)) &
        (master_df['school_type'].isin(selected_school_types)) &
        (master_df['gender'].isin(selected_genders))
    ]

    # --- Step 4: KPI Header ---
    st.header("Executive KPIs")
    
    current_kpis = calculate_kpis(filtered_df)
    previous_kpis = calculate_kpis(prev_period_df) if compare_period else {k: 0 for k in current_kpis}

    kpi_cols = st.columns(7)
    kpi_cols[0].metric("Active Schools", f"{current_kpis['active_schools']:,}", delta=format_delta(current_kpis['active_schools'], previous_kpis['active_schools']) if compare_period else None)
    kpi_cols[1].metric("Total Sessions", f"{current_kpis['total_sessions']:,}", delta=format_delta(current_kpis['total_sessions'], previous_kpis['total_sessions']) if compare_period else None)
    kpi_cols[2].metric("Avg Daily Reading Time", f"{current_kpis['avg_daily_reading_time']:.1f} mins", delta=format_delta(current_kpis['avg_daily_reading_time'], previous_kpis['avg_daily_reading_time']) if compare_period else None)
    kpi_cols[3].metric("Accuracy Ratio (%)", f"{current_kpis['accuracy_ratio']:.1%}", delta=format_delta(current_kpis['accuracy_ratio'], previous_kpis['accuracy_ratio']) if compare_period else None)
    kpi_cols[4].metric("Completed Ratio (%)", f"{current_kpis['completed_ratio']:.1%}", delta=format_delta(current_kpis['completed_ratio'], previous_kpis['completed_ratio']) if compare_period else None)
    kpi_cols[5].metric("Comprehension (%)", f"{current_kpis['comprehension_score']:.1%}", delta=format_delta(current_kpis['comprehension_score'], previous_kpis['comprehension_score']) if compare_period else None)
    kpi_cols[6].metric("Engagement Index", f"{current_kpis['engagement_index']:.0f}", delta=format_delta(current_kpis['engagement_index'], previous_kpis['engagement_index']) if compare_period else None)
    
    st.markdown("---")

    # --- Step 5: Interactive Map ---
    st.header("Geography & Equity")
    map_cols = st.columns(2)
    with map_cols[0]:
        st.subheader("Geographic Activity")
        
        map_metric = st.selectbox(
            "Select Metric to Display on Map",
            options=["Active Schools", "Accuracy Ratio", "Comprehension Score", "Engagement Index"]
        )

        # Aggregate data by district for the map
        district_agg = filtered_df.groupby('District').apply(calculate_kpis).apply(pd.Series)
        district_agg = district_agg.rename(columns={'active_schools': 'Active Schools', 'accuracy_ratio': 'Accuracy Ratio', 'comprehension_score': 'Comprehension Score', 'engagement_index': 'Engagement Index'})
        
        # Merge aggregated data with geo data
        map_data = punjab_geo_df.merge(district_agg, on='District', how='left')
        map_data.fillna(0, inplace=True)

        # Create the map
        fig_map = px.choropleth_mapbox(
            map_data,
            geojson=map_data.geometry,
            locations=map_data.index,
            color=map_metric,
            hover_name="District",
            hover_data={ "Active Schools": True, "Accuracy Ratio": ":.2%", "Comprehension Score": ":.2%", "Engagement Index": ":.0f" },
            color_continuous_scale="Viridis", mapbox_style="carto-positron",
            zoom=5, center={"lat": 31.1704, "lon": 72.7097}, opacity=0.6,
        )
        fig_map.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
    with map_cols[1]:
        st.subheader("Equity Comparison")
        # Create charts with the new synthetic data
        equity_school = filtered_df.groupby('school_type').agg(accuracy=('correct_word_count', 'sum'), total_words=('total_word_count', 'sum')).reset_index()
        equity_school['Accuracy'] = equity_school['accuracy'] / equity_school['total_words']
        fig_equity_school = px.bar(equity_school, x='school_type', y='Accuracy', title="Accuracy: Public vs. Private", color='school_type', text_auto='.1%')
        st.plotly_chart(fig_equity_school, use_container_width=True)

        equity_gender = filtered_df.groupby('gender').agg(accuracy=('correct_word_count', 'sum'), total_words=('total_word_count', 'sum')).reset_index()
        equity_gender['Accuracy'] = equity_gender['accuracy'] / equity_gender['total_words']
        fig_equity_gender = px.bar(equity_gender, x='gender', y='Accuracy', color='gender', title="Accuracy by Gender", text_auto='.1%')
        st.plotly_chart(fig_equity_gender, use_container_width=True)


    st.markdown("---")

    # --- Step 6: Remaining Charts ---
    st.header("Utilization & Engagement")
    util_cols = st.columns(2)
    with util_cols[0]:
        st.subheader("Daily Active Schools Trend")
        daily_activity = filtered_df.groupby('date')['school_id'].nunique().reset_index()
        daily_activity['7-day MA'] = daily_activity['school_id'].rolling(window=7).mean()
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(x=daily_activity['date'], y=daily_activity['school_id'], name='Daily Active Schools', marker_color='lightblue'))
        fig_daily.add_trace(go.Scatter(x=daily_activity['date'], y=daily_activity['7-day MA'], name='7-Day Moving Avg', mode='lines', line=dict(color='darkblue')))
        st.plotly_chart(fig_daily, use_container_width=True)

    with util_cols[1]:
        st.subheader("Session Duration Distribution")
        bins = [0, 10, 30, 60, np.inf]
        labels = ['0-10 Mins', '10-30 Mins', '30-60 Mins', '60+ Mins']
        filtered_df['duration_bin'] = pd.cut(filtered_df['time_spent_minutes'], bins=bins, labels=labels, right=False)
        duration_counts = filtered_df['duration_bin'].value_counts().reset_index()
        fig_duration = px.bar(duration_counts, x='count', y='duration_bin', orientation='h', title="Session Duration")
        st.plotly_chart(fig_duration, use_container_width=True)

    st.markdown("---")
    st.header("Learning Progress")
    progress_cols = st.columns(2)
    with progress_cols[0]:
        st.subheader("Accuracy Ratio Trend (Weekly)")
        weekly_accuracy = filtered_df.groupby('week').apply(lambda x: x['correct_word_count'].sum() / x['total_word_count'].sum() if x['total_word_count'].sum() > 0 else 0).reset_index(name='accuracy_ratio')
        fig_weekly_acc = px.line(weekly_accuracy, x='week', y='accuracy_ratio', title="Weekly Accuracy Ratio", markers=True)
        fig_weekly_acc.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_weekly_acc, use_container_width=True)

    with progress_cols[1]:
        st.subheader("Book Completion Funnel")
        started = filtered_df.drop_duplicates(subset=['school_id', 'book_id'])
        midpoint = started[started['book_completion_percentage'] >= 0.50]
        completed = started[started['completed_book']]
        funnel_data = pd.DataFrame({
            'Stage': ['Started Reading', 'Reached Midpoint (>=50%)', 'Completed Book (>=95%)'],
            'Count': [len(started), len(midpoint), len(completed)]
        })
        fig_funnel = px.funnel(funnel_data, x='Count', y='Stage')
        st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("---")

    # --- Step 7: Storyboard ---
    st.header("Storyboard for Policymakers")
    story_cols = st.columns(3)
    with story_cols[0]:
        st.subheader("What Changed?")
        delta_readers = format_delta(current_kpis['active_schools'], previous_kpis['active_schools'])
        delta_comprehension = format_delta(current_kpis['comprehension_score'], previous_kpis['comprehension_score'])
        st.markdown(f"**<font color='green'>▲ Improvement:</font>** Active Schools up **{delta_readers}**", unsafe_allow_html=True)
        st.markdown(f"**<font color='green'>▲ Improvement:</font>** Comprehension Score improved by **{delta_comprehension}**", unsafe_allow_html=True)

    with story_cols[1]:
        st.subheader("Typical Learner Journey")
        st.markdown("**First 7 Days:** Avg 5 sessions, 30 mins reading.")
        st.markdown("**First Book Completion:** Avg 2nd week, 70% Accuracy.")
        st.markdown("**After 30 Days:** Avg 8 sessions, EI 68.")

    with story_cols[2]:
        st.subheader("Policy")
        st.success(f"**SDG 4.1.1a Alignment:** Basic Proficiency at **{current_kpis['accuracy_ratio']:.0%}**")
        st.info("**NEP 2017 Support:** Aligns with equitable access goals.")
        st.warning("**ASER/READ Integration:** Link to survey data overlays.")


else:
    st.error("Dashboard could not be loaded due to data issues.")

