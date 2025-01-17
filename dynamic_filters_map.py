import os
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import sqlite3
from states_json_util import find_state
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import geopandas as gpd

st.set_page_config(layout="wide", page_title="Dynamic Railroad Incident Map")

# CSS for dark background
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E; 
    }
    section[data-testid="stSidebar"] {
        background-color: #2E2E2E; 
    }
    body, .markdown-text-container, .sidebar .sidebar-content {
        color: #FFFFFF;
    }
    .css-1d391kg, .css-1aumxhk, .css-1b0td9e {
        background-color: #3E3E3E;
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

# Theme toggle
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

def toggle_theme():
    st.session_state["theme"] = (
        "light" 
        if (st.session_state["theme"] == "dark" or st.session_state["theme"] == "states") 
        else "dark"
    )

if st.sidebar.button("Toggle Dark/Light Theme"):
    toggle_theme()

map_tiles = {
    "dark": "CartoDB dark_matter",
    "light": "CartoDB positron",
    "states": "CartoDB positron"
}.get(st.session_state["theme"], "CartoDB dark_matter")

# Path to your SQLite DB
sqlite_db = r"C:\\dev\\visgg\\data\\railroad_incidents_cleanedMUT.db"
state_borders_df = gpd.read_file("C:\\dev\\visgg\\data\\us-states.json")
state_borders_df = state_borders_df.to_crs(epsg=4326)

def update_overlay(should_update):
    if should_update:
        for _, r in state_borders_df.iterrows():
            # Without simplifying the representation of each borough,
            # the map might not be displayed
            sim_geo = gpd.GeoSeries(r["geometry"]).simplify(tolerance=0.001)
            geo_j = sim_geo.to_json()
            geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {"fillColor": "orange"})
            # folium.Popup(r["BoroName"]).add_to(geo_j)
            geo_j.add_to(m)
        
actual_name = {
    # Speed Categories
    "Very Slow (Speed ≤ 20)": "Very Slow",
    "Slow (Speed between 21 and 40)": "Slow",
    "Moderate (Speed between 41 and 60)": "Moderate",
    "Fast (Speed between 61 and 80)": "Fast",
    "High Speed (Speed > 80)": "High Speed",

    # Year Categories
    "Before 1990 (Year < 1990)": "Before 1990",
    "1990-1999 (Year between 1990 and 1999)": "1990-1999",
    "2000-2009 (Year between 2000 and 2009)": "2000-2009",
    "2010-2019 (Year between 2010 and 2019)": "2010-2019",
    "2020 and Later (Year >= 2020)": "2020 and Later",

    # Weather Conditions
    "Clear (1)": "Clear",
    "Cloudy (2)": "Cloudy",
    "Rain (3)": "Rain",
    "Fog (4)": "Fog",
    "Snow (5)": "Snow",

    # Equipment Damage Categories
    "Minimal (Damage ≤ $10,000)": "Minimal",
    "Moderate (Damage between $10,001 and $100,000)": "Moderate",
    "Significant (Damage between $100,001 and $1,000,000)": "Significant",
    "Severe (Damage > $1,000,000)": "Severe",

    # Track Damage Categories
    "Minimal (Damage ≤ $10,000)": "Minimal",
    "Moderate (Damage between $10,001 and $100,000)": "Moderate",
    "Significant (Damage between $100,001 and $1,000,000)": "Significant",
    "Severe (Damage > $1,000,000)": "Severe",

    # Death Categories
    "No Deaths (0 deaths)": "No Deaths",
    "Isolated (1 death)": "Isolated",
    "Moderate Fatalities (2-10 deaths)": "Moderate Fatalities",
    "High Fatalities (11-50 deaths)": "High Fatalities",
    "Catastrophic (> 50 deaths)": "Catastrophic",

    # Injury Categories
    "No Injuries (0 injuries)": "No Injuries",
    "Low Severity (1-10 injuries)": "Low Severity",
    "Moderate Severity (11-50 injuries)": "Moderate Severity",
    "High Severity (51-100 injuries)": "High Severity",
    "Catastrophic (> 100 injuries)": "Catastrophic"
}
@st.cache_data
def get_filter_options(table_name, column_name):
    conn = sqlite3.connect(sqlite_db)
    query = f"SELECT DISTINCT {column_name} FROM {table_name};"
    options = [row[0] for row in conn.execute(query).fetchall()]
    conn.close()
    return options

# Now we also fetch possible year group values from `Year_Groups` table
# speed_categories = get_filter_options("Train_Speed_Categories", "Speed_Category")
# weather_conditions = get_filter_options("Weather_Conditions", "Weather_Condition")
# death_categories = get_filter_options("Death_Categories", "Death_Category")
# injury_categories = get_filter_options("Injury_Categories", "Injury_Category")
# damage_categories = get_filter_options("Equipment_Damage_Categories", "Damage_Category")
# year_group_categories = get_filter_options("Year_Groups", "Year_Group")  # <-- new

speed_categories = [
    "Very Slow (Speed ≤ 20)",
    "Slow (Speed between 21 and 40)",
    "Moderate (Speed between 41 and 60)",
    "Fast (Speed between 61 and 80)",
    "High Speed (Speed > 80)"
]

# Year Categories
year_group_categories = [
    "Before 1990 (Year < 1990)",
    "1990-1999 (Year between 1990 and 1999)",
    "2000-2009 (Year between 2000 and 2009)",
    "2010-2019 (Year between 2010 and 2019)",
    "2020 and Later (Year >= 2020)"
]

# Weather Conditions
weather_conditions = [
    "Clear (1)",
    "Cloudy (2)",
    "Rain (3)",
    "Fog (4)",
    "Snow (5)"
]

# Equipment Damage Categories
damage_categories = [
    "Minimal (Damage ≤ $10,000)",
    "Moderate (Damage between $10,001 and $100,000)",
    "Significant (Damage between $100,001 and $1,000,000)",
    "Severe (Damage > $1,000,000)"
]

# Track Damage Categories
track_damage_categories = [
    "Minimal (Damage ≤ $10,000)",
    "Moderate (Damage between $10,001 and $100,000)",
    "Significant (Damage between $100,001 and $1,000,000)",
    "Severe (Damage > $1,000,000)"
]

# Death Categories
death_categories = [
    "No Deaths (0 deaths)",
    "Isolated (1 death)",
    "Moderate Fatalities (2-10 deaths)",
    "High Fatalities (11-50 deaths)",
    "Catastrophic (> 50 deaths)"
]

# Injury Categories
injury_categories = [
    "No Injuries (0 injuries)",
    "Low Severity (1-10 injuries)",
    "Moderate Severity (11-50 injuries)",
    "High Severity (51-100 injuries)",
    "Catastrophic (> 100 injuries)"
]


# Sidebar: Visualization Mode
st.sidebar.title("Visualization Mode")
visualization_mode = st.sidebar.selectbox(
    "Select Visualization",
    ["Multi-Scatter Plots", "Radar Plot", "Line Chart", "Bar Chart"]
)

# "Incident" or "State" mode
if "mode" not in st.session_state:
    st.session_state["mode"] = "Incident Details"

def switch_mode(new_mode):
    st.session_state["mode"] = new_mode

mode_options = ["Incident Details", "State Details"]
selected_mode = st.sidebar.selectbox(
    "Select Mode", mode_options, index=mode_options.index(st.session_state["mode"])
)
if selected_mode != st.session_state["mode"]:
    switch_mode(selected_mode)

if st.session_state["mode"] == "State Details":
    st.sidebar.title("State Details")
    summary_container = st.sidebar.container()
else:
    st.sidebar.title("Incident Details")
    summary_container = st.sidebar.container()

# Dynamic Filters
st.sidebar.title("Dynamic Filters")
col1, col2 = st.sidebar.columns(2)

with col1:
    selected_speed = st.selectbox("Speed Category", ["All"] + speed_categories)
    selected_weather = st.selectbox("Weather Condition", ["All"] + weather_conditions)
    selected_year_group = st.selectbox("Year Group", ["All"] + year_group_categories)

with col2:
    selected_death = st.selectbox("Death Category", ["All"] + death_categories)
    selected_injury = st.selectbox("Injury Category", ["All"] + injury_categories)
    selected_damage = st.selectbox("Damage Category", ["All"] + damage_categories)

enable_clustering = st.sidebar.checkbox("Enable Clustering", value=True)
# show_heatmap = st.sidebar.checkbox("Show Heatmap")

query = """
SELECT 
    R.latitude,
    R.longitud,
    R.trnspd,
    R.eqpdmg,
    R.trkdmg,
    R.caskld,
    R.casinj,
    R.year,
    R.month,
    C.Accident_Type,
    R.narr1, R.narr2, R.narr3, R.narr4, R.narr5, 
    R.narr6, R.narr7, R.narr8, R.narr9, R.narr10,
    R.narr11, R.narr12, R.narr13, R.narr14, R.narr15,
    R.state_name,
    W.Weather_Condition,
    YG.Year_Group   -- We'll select the Year_Group from your Year_Groups table
FROM Railroad_Incidents R
LEFT JOIN Categorized_Incidents_By_ID C ON R.ID = C.ID
LEFT JOIN Train_Speed_Categories S ON R.ID = S.ID
LEFT JOIN Weather_Conditions W ON R.ID = W.ID
LEFT JOIN Death_Categories D ON R.ID = D.ID
LEFT JOIN Injury_Categories I ON R.ID = I.ID
LEFT JOIN Equipment_Damage_Categories E ON R.ID = E.ID
LEFT JOIN Year_Groups YG ON R.ID = YG.ID   -- new join
WHERE 1=1
"""

if selected_speed != "All":
    query += f" AND S.Speed_Category = '{actual_name[selected_speed]}'"
if selected_weather != "All":
    query += f" AND W.Weather_Condition = '{actual_name[selected_weather]}'"

if selected_death != "All":
    query += f" AND D.Death_Category = '{actual_name[selected_death]}'"
if selected_injury != "All":
    query += f" AND I.Injury_Category = '{actual_name[selected_injury]}'"
if selected_damage != "All":
    query += f" AND E.Damage_Category = '{actual_name[selected_damage]}'"

if selected_year_group != "All":
    query += f" AND YG.Year_Group = '{actual_name[selected_year_group]}'"

query += " LIMIT 500;"

@st.cache_data
def do_db_query(sql_query):
    conn = sqlite3.connect(sqlite_db)
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df

filtered_data = do_db_query(query)

#Convert numeric columns
for col in ["trnspd", "eqpdmg", "trkdmg", "caskld", "casinj", "year", "month"]:
    filtered_data[col] = pd.to_numeric(filtered_data[col], errors="coerce")

#Combine narrative columns
def combine_narratives(row):
    return " ".join(
        str(row[f"narr{i}"])
        for i in range(1,16)
        if pd.notnull(row[f"narr{i}"])
    )

filtered_data["description"] = filtered_data.apply(combine_narratives, axis=1)

if filtered_data.empty:
    st.warning("No data available for the selected filters (including Year_Group).")
    st.stop()

#Color dimension
map_color_dimension = st.sidebar.selectbox(
    "Color Map Markers By",
    ["Accident_Type", "Weather_Condition"],
    index=0
)

def build_unified_color_map(df, color_dimension):
    unique_vals = df[color_dimension].dropna().unique()
    color_palette = px.colors.qualitative.Plotly
    return {
        val: color_palette[i % len(color_palette)]
        for i, val in enumerate(unique_vals)
    }

color_map = build_unified_color_map(filtered_data, map_color_dimension)

#Session states
if "selected_incident" not in st.session_state:
    st.session_state["selected_incident"] = None
if "clicked_state" not in st.session_state:
    st.session_state["clicked_state"] = None

def build_map_center():
    if st.session_state["selected_incident"] is not None:
        return [st.session_state["selected_incident"][0], st.session_state["selected_incident"][1]], 13
    else:
        return [37.0902, -95.7129], 5

map_center, map_zoom = build_map_center()
m = folium.Map(location=map_center, zoom_start=map_zoom, tiles=map_tiles)
update_overlay(st.session_state["mode"] == "State Details")

if st.session_state["mode"] == "State Details" and st.session_state["clicked_state"]:
    title_html = f"""
    <div style="position: fixed;
                top: 70px; left: 50px; width: 220px;
                z-index:9999;
                font-size:18px;
                background-color: rgba(0, 0, 0, 0.5);
                padding: 8px;
                border-radius: 5px;">
        <span style="color: #fff;">In {st.session_state["clicked_state"]}</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

#Add Markers
if st.session_state["mode"] == "State Details":
    if enable_clustering:
        marker_cluster = MarkerCluster().add_to(m)
    else:
        marker_cluster = m

    
else:
    if enable_clustering:
        marker_cluster = MarkerCluster().add_to(m)
    else:
        marker_cluster = m

    folium_color_names = [
        "red","blue","green","purple","orange","darkred","lightred","beige",
        "darkblue","darkgreen","cadetblue","darkpurple","white","pink","lightblue",
        "lightgreen","gray","black","lightgray"
    ]
    unique_hexes = list(color_map.values())

    for _, row in filtered_data.iterrows():
        if pd.isnull(row["latitude"]) or pd.isnull(row["longitud"]):
            continue
        cat_val = row[map_color_dimension]
        hex_color = color_map.get(cat_val, "#808080")
        if hex_color in unique_hexes:
            color_index = unique_hexes.index(hex_color)
            folium_color = folium_color_names[color_index % len(folium_color_names)]
        else:
            folium_color = "gray"

        folium.Marker(
            location=[row["latitude"], row["longitud"]],
            icon=folium.Icon(color=folium_color),
            tooltip="Click for details"
        ).add_to(marker_cluster)

# if show_heatmap:
#     heat_data = filtered_data[["latitude", "longitud"]].dropna().values.tolist()
#     if heat_data:
#         HeatMap(heat_data).add_to(m)

clicked_data = st_folium(m, use_container_width=True, height=600)

#Sidebar info
with summary_container:
    if clicked_data and clicked_data.get("last_clicked"):
        lat = clicked_data["last_clicked"]["lat"]
        lon = clicked_data["last_clicked"]["lng"]

        if st.session_state["mode"] == "State Details":
            clickedState = find_state(lat, lon)
            st.session_state["clicked_state"] = str(clickedState)
            st.write(f"### Clicked State Coordinates: ({lat}, {lon})")
            st.write(f"### Clicked State: {clickedState}")

            state_data = filtered_data[filtered_data['state_name'] == clickedState]
            if state_data.empty:
                st.warning(f"No data available for state: {clickedState}")
            else:
                total_incidents = len(state_data)
                st.write(f"**Total incidents in {clickedState}:** {total_incidents}")

                if "trnspd" in state_data and state_data["trnspd"].notnull().any():
                    st.write(f"**Avg Speed:** {state_data['trnspd'].mean():.2f}")
                else:
                    st.write("Avg Speed not available.")

                total_fatalities = state_data["caskld"].sum()
                total_injuries = state_data["casinj"].sum()
                st.write(f"**Total Fatalities:** {int(total_fatalities)}")
                st.write(f"**Total Injuries:** {int(total_injuries)}")

                if "Weather_Condition" in state_data and state_data["Weather_Condition"].notnull().any():
                    common_weather = state_data["Weather_Condition"].value_counts().idxmax()
                    st.write(f"**Most common weather:** {common_weather}")
                else:
                    st.write("No weather data.")

                breakdown = state_data["Accident_Type"].value_counts()
                st.write("**Incident breakdown:**")
                st.write(breakdown)
        else:
            st.session_state["selected_incident"] = (lat, lon)
            incident = filtered_data[(filtered_data["latitude"] == lat) & (filtered_data["longitud"] == lon)]
            if not incident.empty:
                r = incident.iloc[0]
                st.write("### Incident Details")
                st.write(f"**Accident Type:** {r['Accident_Type']}")
                st.write(f"**Injuries:** {r['casinj']}")
                st.write(f"**Deaths:** {r['caskld']}")
                st.write(f"**Description:** {r['description']}")
                st.write(f"**State:** {r['state_name']}")
                st.write(f"**Train Speed:** {r['trnspd']} mph")
                st.write(f"**Equipment Damage:** ${r['eqpdmg']}")
                st.write(f"**Track Damage:** ${r['trkdmg']}")
                st.write(f"**Weather Condition:** {r['Weather_Condition']}")
                # Show the Year_Group if you like
                st.write(f"**Year Group:** {r['Year_Group']}")
            else:
                st.warning("No incident data found for that location.")

    elif clicked_data and clicked_data.get("last_object_clicked") and st.session_state["mode"] == "Incident Details":
        lat = clicked_data["last_object_clicked"]["lat"]
        lon = clicked_data["last_object_clicked"]["lng"]
        st.session_state["selected_incident"] = (lat, lon)
        incident = filtered_data[(filtered_data["latitude"] == lat) & (filtered_data["longitud"] == lon)]
        if not incident.empty:
            r = incident.iloc[0]
            st.write("### Incident Details")
            st.write(f"**Accident Type:** {r['Accident_Type']}")
            st.write(f"**Injuries:** {r['casinj']}")
            st.write(f"**Deaths:** {r['caskld']}")
            st.write(f"**Description:** {r['description']}")
            st.write(f"**State:** {r['state_name']}")
            st.write(f"**Train Speed:** {r['trnspd']} mph")
            st.write(f"**Equipment Damage:** ${r['eqpdmg']}")
            st.write(f"**Track Damage:** ${r['trkdmg']}")
            st.write(f"**Weather Condition:** {r['Weather_Condition']}")
            st.write(f"**Year Group:** {r['Year_Group']}")
        else:
            st.warning("No incident data found for that location.")

#Visualization Modes
#1) Multi-Scatter Plots
if visualization_mode == "Multi-Scatter Plots":
    st.markdown("## Multi-Scatter Plots: Damage, Fatalities, Injuries vs. Speed")

    #If in State mode and a state is selected, limit data to that state
    if st.session_state["mode"] == "State Details" and st.session_state["clicked_state"]:
        scatter_title_suffix = f" in {st.session_state['clicked_state']}"
        chart_data = filtered_data[filtered_data['state_name'] == st.session_state["clicked_state"]]
    else:
        scatter_title_suffix = ""
        chart_data = filtered_data.copy()

    if chart_data.empty:
        st.warning("No data available for the selected filters.")
    else:
        needed_cols = ["trnspd", "eqpdmg", "trkdmg", "caskld", "casinj",
                       "latitude", "longitud", map_color_dimension, "description"]
        chart_data = chart_data.dropna(subset=needed_cols)
        if chart_data.empty:
            st.warning("No data to plot on the scatter charts with the current selections.")
        else:
            #Build 2x2 subplots
            fig = make_subplots(
                rows=2, cols=2, 
                subplot_titles=(
                    "Equipment Damage vs Speed", 
                    "Track Damage vs Speed",
                    "Fatalities vs Speed",
                    "Injuries vs Speed"
                ),
                shared_xaxes=False,
                shared_yaxes=False
            )

            #Collect categories for legend
            used_categories = set()
            unique_categories = chart_data[map_color_dimension].unique()

            for cat_val in unique_categories:
                subdf = chart_data[chart_data[map_color_dimension] == cat_val]
                cat_color = color_map.get(cat_val, "#808080")

                show_legend = cat_val not in used_categories

                # Equipment vs Speed
                eqp_trace = go.Scatter(
                    x=subdf["trnspd"],
                    y=subdf["eqpdmg"],
                    mode="markers",
                    marker=dict(color=cat_color, size=7),
                    text=subdf["description"],
                    hovertemplate=(
                        "Speed: %{x} mph<br>"
                        "Equipment Damage: $%{y}<br>"
                        f"{map_color_dimension}: {cat_val}<br>"
                        "Lat: %{meta[0]}<br>"
                        "Lon: %{meta[1]}<extra></extra>"
                    ),
                    customdata=[cat_val]*len(subdf),
                    meta=list(zip(subdf["latitude"], subdf["longitud"])),
                    name=str(cat_val),
                    legendgroup=str(cat_val),
                    showlegend=False
                )
                fig.add_trace(eqp_trace, row=1, col=1)

                # Track vs Speed
                trk_trace = go.Scatter(
                    x=subdf["trnspd"],
                    y=subdf["trkdmg"],
                    mode="markers",
                    marker=dict(color=cat_color, size=7),
                    text=subdf["description"],
                    hovertemplate=(
                        "Speed: %{x} mph<br>"
                        "Track Damage: $%{y}<br>"
                        f"{map_color_dimension}: {cat_val}<br>"
                        "Lat: %{meta[0]}<br>"
                        "Lon: %{meta[1]}<extra></extra>"
                    ),
                    customdata=[cat_val]*len(subdf),
                    meta=list(zip(subdf["latitude"], subdf["longitud"])),
                    name=str(cat_val),
                    legendgroup=str(cat_val),
                    showlegend=False
                )
                fig.add_trace(trk_trace, row=1, col=2)

                # Fatalities vs Speed
                fat_trace = go.Scatter(
                    x=subdf["trnspd"],
                    y=subdf["caskld"],
                    mode="markers",
                    marker=dict(color=cat_color, size=7),
                    text=subdf["description"],
                    hovertemplate=(
                        "Speed: %{x} mph<br>"
                        "Fatalities: %{y}<br>"
                        f"{map_color_dimension}: {cat_val}<br>"
                        "Lat: %{meta[0]}<br>"
                        "Lon: %{meta[1]}<extra></extra>"
                    ),
                    customdata=[cat_val]*len(subdf),
                    meta=list(zip(subdf["latitude"], subdf["longitud"])),
                    name=str(cat_val),
                    legendgroup=str(cat_val),
                    showlegend=False
                )
                fig.add_trace(fat_trace, row=2, col=1)

                # Injuries vs Speed
                inj_trace = go.Scatter(
                    x=subdf["trnspd"],
                    y=subdf["casinj"],
                    mode="markers",
                    marker=dict(color=cat_color, size=7),
                    text=subdf["description"],
                    hovertemplate=(
                        "Speed: %{x} mph<br>"
                        "Injuries: %{y}<br>"
                        f"{map_color_dimension}: {cat_val}<br>"
                        "Lat: %{meta[0]}<br>"
                        "Lon: %{meta[1]}<extra></extra>"
                    ),
                    customdata=[cat_val]*len(subdf),
                    meta=list(zip(subdf["latitude"], subdf["longitud"])),
                    name=str(cat_val),
                    legendgroup=str(cat_val),
                    showlegend=show_legend
                )
                fig.add_trace(inj_trace, row=2, col=2)

                # Update used categories after adding all traces for this category
                used_categories.add(cat_val)

            #Highlight selected incident if any
            if st.session_state["selected_incident"] is not None:
                hilat, hilon = st.session_state["selected_incident"]
                row_match = chart_data[
                    (chart_data["latitude"] == hilat) & (chart_data["longitud"] == hilon)
                ]
                if not row_match.empty:
                    r = row_match.iloc[0]
                    highlight_x_eqp = r["trnspd"]
                    highlight_y_eqp = r["eqpdmg"]
                    highlight_x_trk = r["trnspd"]
                    highlight_y_trk = r["trkdmg"]
                    highlight_x_fat = r["trnspd"]
                    highlight_y_fat = r["caskld"]
                    highlight_x_inj = r["trnspd"]
                    highlight_y_inj = r["casinj"]

                    # star markers
                    fig.add_trace(go.Scatter(
                        x=[highlight_x_eqp],
                        y=[highlight_y_eqp],
                        mode="markers",
                        marker=dict(size=15, symbol="star", color="yellow"),
                        name="Selected Incident",
                        showlegend=False
                    ), row=1, col=1)

                    fig.add_trace(go.Scatter(
                        x=[highlight_x_trk],
                        y=[highlight_y_trk],
                        mode="markers",
                        marker=dict(size=15, symbol="star", color="yellow"),
                        name="Selected Incident",
                        showlegend=False
                    ), row=1, col=2)

                    fig.add_trace(go.Scatter(
                        x=[highlight_x_fat],
                        y=[highlight_y_fat],
                        mode="markers",
                        marker=dict(size=15, symbol="star", color="yellow"),
                        name="Selected Incident",
                        showlegend=False
                    ), row=2, col=1)

                    fig.add_trace(go.Scatter(
                        x=[highlight_x_inj],
                        y=[highlight_y_inj],
                        mode="markers",
                        marker=dict(size=15, symbol="star", color="yellow"),
                        name="Selected Incident",
                        showlegend=False
                    ), row=2, col=2)

            #Update axes titles
            fig.update_xaxes(title_text="Train Speed (mph)", row=1, col=1)
            fig.update_yaxes(title_text="Equipment Damage ($)", row=1, col=1)
            fig.update_xaxes(title_text="Train Speed (mph)", row=1, col=2)
            fig.update_yaxes(title_text="Track Damage ($)", row=1, col=2)
            fig.update_xaxes(title_text="Train Speed (mph)", row=2, col=1)
            fig.update_yaxes(title_text="Killed", row=2, col=1)
            fig.update_xaxes(title_text="Train Speed (mph)", row=2, col=2)
            fig.update_yaxes(title_text="Injuries", row=2, col=2)

            fig.update_layout(
                title_text=f"Damage, Fatalities & Injuries vs. Speed{scatter_title_suffix}",
                plot_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                paper_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                font_color='#FFFFFF' if st.session_state["theme"] == "dark" else '#000000',
                hovermode="closest",
                clickmode="event+select",
                legend=dict(
                    x=1.02, y=1.0,
                    xanchor="left", yanchor="top",
                    title=f"{map_color_dimension}"
                )
            )

            st.markdown("**Click any point to zoom the map to that incident.**")

            #plotly_events to capture scatter clicks
            selected_points = plotly_events(
                fig,
                click_event=True,
                hover_event=False,
                select_event=False,
                override_height=800,
                override_width="100%"
            )

            #If user clicked a point, center the map at that incident
            if selected_points:
                point_index = selected_points[0]["pointIndex"]
                trace_index = selected_points[0]["curveNumber"]
                lat_val = float(fig.data[trace_index].meta[point_index][0])
                lon_val = float(fig.data[trace_index].meta[point_index][1])
                st.session_state["selected_incident"] = (lat_val, lon_val)
                #If in State Mode, switch to Incident mode
                if st.session_state["mode"] == "State Details":
                    st.session_state["mode"] = "Incident Details"
                st.info(f"Switched to 'Incident Details' mode and centered map at: (lat={lat_val}, lon={lon_val}).")

            #Button to clear incident filter
            if st.button("Clear Incident Filter"):
                st.session_state["selected_incident"] = None
                st.info("Incident filter cleared. Map is reset to the default view.")

#2) Radar Plot
elif visualization_mode == "Radar Plot":
    st.markdown("## Radar Plot")

    #2A. State-based radar (if in State Mode)
    if st.session_state["mode"] == "State Details":
        st.markdown("#### State Radar Plot (Comparison)")
        if st.session_state["clicked_state"]:
            base_state = st.session_state["clicked_state"]
            base_data = filtered_data[filtered_data['state_name'] == base_state]

            #Pick a comparison state
            all_states = sorted(filtered_data['state_name'].dropna().unique())
            comparison_state = st.sidebar.selectbox(
                "Compare with another State",
                ["None"] + all_states,
                index=0
            )

            import numpy as np

            def compute_radar_values_state(state_df):
                """Compute normalized (0..1) radar values for the given state."""
                if state_df.empty:
                    return None

                attrs = {
                    "trnspd": "Average Speed (mph)",
                    "trkdmg": "Track Damage ($)",
                    "caskld": "Deaths",
                    "casinj": "Injuries",
                    "eqpdmg": "Equipment Damage ($)"
                }
                normalized_values = {}

                for col in attrs.keys():
                    col_data = state_df[col].dropna()
                    if col_data.empty:
                        normalized_values[col] = 0.0
                        continue

                    #SHIFT + LOG + MIN–MAX 
                    min_val_raw = col_data.min()
                    shift_amount = 1 - min_val_raw if min_val_raw < 1 else 0
                    col_data_shifted = col_data + shift_amount

                    col_data_log = np.log(col_data_shifted)
                    col_log_min, col_log_max = col_data_log.min(), col_data_log.max()

                    if col_log_min == col_log_max:
                        col_norm_mean = 0.0
                    else:
                        col_data_norm = (col_data_log - col_log_min) / (col_log_max - col_log_min)
                        col_norm_mean = col_data_norm.mean()

                    normalized_values[col] = col_norm_mean

                return normalized_values

            base_radar_dict = compute_radar_values_state(base_data)

            if not base_radar_dict:
                st.warning(f"No data available for {base_state} to plot the Radar Chart.")
            else:
                attrs_order = ["trnspd", "trkdmg", "caskld", "casinj", "eqpdmg"]
                attr_labels = [
                    "Average Speed (mph)",
                    "Track Damage ($)",
                    "Deaths",
                    "Injuries",
                    "Equipment Damage ($)"
                ]
                base_radar_values = [base_radar_dict[a] for a in attrs_order]

                fig = go.Figure()
                #Add base state
                fig.add_trace(go.Scatterpolar(
                    r=base_radar_values,
                    theta=attr_labels,
                    fill='toself',
                    fillcolor='rgba(0, 180, 255, 0.3)',
                    line_color='rgba(0, 180, 255, 1)',
                    marker=dict(symbol='circle', size=6, color='rgba(0, 180, 255, 1)'),
                    name=f"{base_state}",
                    hovertemplate="<b>%{theta}</b>: %{r:.2f}<extra></extra>"
                ))

                #If user chose a comparison state
                if comparison_state != "None":
                    comp_data = filtered_data[filtered_data['state_name'] == comparison_state]
                    comp_radar_dict = compute_radar_values_state(comp_data)
                    if comp_radar_dict:
                        comp_radar_values = [comp_radar_dict[a] for a in attrs_order]
                        fig.add_trace(go.Scatterpolar(
                            r=comp_radar_values,
                            theta=attr_labels,
                            fill='toself',
                            fillcolor='rgba(255, 100, 0, 0.3)',
                            line_color='rgba(255, 100, 0, 1)',
                            marker=dict(symbol='circle', size=6, color='rgba(255, 100, 0, 1)'),
                            name=f"{comparison_state}",
                            hovertemplate="<b>%{theta}</b>: %{r:.2f}<extra></extra>"
                        ))
                    else:
                        st.warning(f"No data found for comparison state: {comparison_state}")

                fig.update_layout(
                    polar=dict(
                        bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                        radialaxis=dict(
                            visible=True,
                            range=[0,1],
                            showline=True,
                            linewidth=2,
                            linecolor='rgba(255,255,255,0.2)' if st.session_state["theme"] == "dark" else 'rgba(0,0,0,0.1)',
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.2)' if st.session_state["theme"] == "dark" else 'rgba(0,0,0,0.1)',
                            gridwidth=1,
                            tickfont=dict(color='#FFFFFF' if st.session_state["theme"] == "dark" else '#000000'),
                            tickvals=[0,0.2,0.4,0.6,0.8,1.0]
                        ),
                        angularaxis=dict(
                            visible=True,
                            showline=True,
                            linewidth=1,
                            linecolor='rgba(255,255,255,0.2)' if st.session_state["theme"] == "dark" else 'rgba(0,0,0,0.1)',
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.2)' if st.session_state["theme"] == "dark" else 'rgba(0,0,0,0.1)',
                            tickfont=dict(color='#FFFFFF' if st.session_state["theme"] == "dark" else '#000000'),
                            rotation=90,
                        ),
                    ),
                    paper_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                    plot_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                    font=dict(color='#FFFFFF' if st.session_state["theme"] == "dark" else '#000000'),
                    showlegend=True,
                    margin=dict(l=60, r=60, t=120, b=60),
                )

                chart_title = f"<b>Balanced Radar Plot</b><br>{base_state}"
                if comparison_state != "None":
                    chart_title += f" vs {comparison_state}"

                fig.update_layout(
                    title=dict(
                        text=chart_title,
                        x=0.5, y=0.95,
                        xanchor='center', yanchor='top',
                        font=dict(size=18)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Click on a state marker in the map to see its Radar Plot.")

    #2B. Incident-based radar (if in Incident Mode)
    if st.session_state["mode"] == "Incident Details":
        st.markdown("#### Incident Radar Plot (Single Incident)")
        if st.session_state["selected_incident"] is not None:
            lat_inc, lon_inc = st.session_state["selected_incident"]
            inc_row = filtered_data[
                (filtered_data["latitude"] == lat_inc) & (filtered_data["longitud"] == lon_inc)
            ]
            if not inc_row.empty:
                rowvals = inc_row.iloc[0]
                attributes = ["trnspd", "eqpdmg", "trkdmg", "caskld", "casinj"]
                display_labels = ["Train Speed", "Equip Damage", "Track Damage", "Deaths", "Injuries"]

                mins = {}
                maxs = {}
                for col in attributes:
                    col_data = filtered_data[col].dropna()
                    if col_data.empty:
                        mins[col] = 0
                        maxs[col] = 1
                    else:
                        mins[col] = col_data.min()
                        maxs[col] = col_data.max()

                #Now build normalized [0..1] for this one incident
                incident_radar_vals = []
                for col in attributes:
                    val = rowvals[col] if not pd.isnull(rowvals[col]) else 0.0
                    rng = maxs[col] - mins[col]
                    if rng == 0:
                        norm_val = 0.0
                    else:
                        norm_val = (val - mins[col]) / rng
                    incident_radar_vals.append(norm_val)

                #Single-incident radar
                fig_incident = go.Figure(data=go.Scatterpolar(
                    r=incident_radar_vals,
                    theta=display_labels,
                    fill='toself',
                    fillcolor='rgba(200, 50, 255, 0.3)',
                    line_color='rgba(200, 50, 255, 1)',
                    marker=dict(symbol='circle', size=6, color='rgba(200, 50, 255, 1)'),
                    name="This Incident"
                ))

                fig_incident.update_layout(
                    polar=dict(
                        bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                        radialaxis=dict(
                            visible=True,
                            range=[0,1],
                            showline=True,
                            linewidth=2,
                            linecolor='rgba(255,255,255,0.2)' if st.session_state["theme"] == "dark" else 'rgba(0,0,0,0.1)',
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.2)' if st.session_state["theme"] == "dark" else 'rgba(0,0,0,0.1)',
                            gridwidth=1,
                            tickfont=dict(color='#FFFFFF' if st.session_state["theme"] == "dark" else '#000000'),
                        ),
                        angularaxis=dict(
                            visible=True,
                            showline=True,
                            linewidth=1,
                            linecolor='rgba(255,255,255,0.2)' if st.session_state["theme"] == "dark" else 'rgba(0,0,0,0.1)',
                            showgrid=True,
                            gridcolor='rgba(255,255,255,0.2)' if st.session_state["theme"] == "dark" else 'rgba(0,0,0,0.1)',
                            tickfont=dict(color='#FFFFFF' if st.session_state["theme"] == "dark" else '#000000'),
                            rotation=90,
                        ),
                    ),
                    paper_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                    plot_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                    font=dict(color='#FFFFFF' if st.session_state["theme"] == "dark" else '#000000'),
                    showlegend=False,
                    margin=dict(l=60, r=60, t=60, b=60),
                    title=f"Radar for Incident @ (lat={lat_inc:.3f}, lon={lon_inc:.3f})"
                )

                st.plotly_chart(fig_incident, use_container_width=True)
            else:
                st.warning("No data found for the selected incident to build a radar chart.")
        else:
            st.warning("Click an incident on the map to see its radar plot.")

#3) Line Chart (NO datetime parsing — group by year/month)
elif visualization_mode == "Line Chart":
    st.markdown("### Line Chart: Incidents Over Time by Year/Month")

    #If we're in State Details mode AND a state is clicked, filter by that state
    if st.session_state["mode"] == "State Details" and st.session_state["clicked_state"]:
        line_data = filtered_data[filtered_data["state_name"] == st.session_state["clicked_state"]]
        suffix = f" in {st.session_state['clicked_state']}"
    else:
        line_data = filtered_data.copy()
        suffix = ""

    #Ensure we have data for the line chart
    line_data = line_data.dropna(subset=["year", "month"])  # we need valid year/month
    if line_data.empty:
        st.warning(f"No data available for the selected filters{suffix}.")
    else:
        #Let user choose if we want "Total Incidents" or "By Weather Condition"
        line_category = st.radio("Line Category", ["Total Incidents", "By Weather Condition"], index=0)

        #Group data by (year, month) or (year, month, weather_condition)
        if line_category == "Total Incidents":
            grouped = line_data.groupby(["year", "month"]).size().reset_index(name="count")
            title_text = f"Total Incidents by Year-Month{suffix}"
        else:
            line_data["Weather_Condition"] = line_data["Weather_Condition"].fillna("Unknown")
            grouped = line_data.groupby(["year", "month", "Weather_Condition"]).size().reset_index(name="count")
            title_text = f"Incidents by Year-Month & Weather Condition{suffix}"

        #Sort for chronological order
        grouped = grouped.sort_values(["year", "month"])
        #Build a "YYYY-MM" label
        grouped["year_month_str"] = grouped.apply(
            lambda r: f"{int(r.year)}-{int(r.month):02d}",
            axis=1
        )

        #Plot with Plotly
        if line_category == "Total Incidents":
            fig = px.line(
                grouped,
                x="year_month_str",
                y="count",
                title=title_text,
                markers=True
            )
            fig.update_traces(line_color="cyan")
        else:
            fig = px.line(
                grouped,
                x="year_month_str",
                y="count",
                color="Weather_Condition",
                title=title_text,
                markers=True
            )

        #Theming
        fig.update_layout(
            xaxis_title="Year-Month",
            yaxis_title="Incident Count",
            plot_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
            paper_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
            font_color='#FFFFFF' if st.session_state["theme"] == "dark" else '#000000'
        )

        st.plotly_chart(fig, use_container_width=True)

# 4) Bar Chart
elif visualization_mode == "Bar Chart":
    st.markdown("### Bar Chart: Incidents by State")

    bar_data = filtered_data.dropna(subset=["state_name"])
    if bar_data.empty:
        st.warning("No data available for the selected filters or no valid state data.")
    else:
        state_counts = bar_data.groupby("state_name").size().reset_index(name="count")
        state_counts = state_counts.sort_values("count", ascending=False)

        fig = px.bar(
            state_counts,
            x="state_name",
            y="count",
            title="Incidents by State",
        )
        fig.update_layout(
            xaxis_title="State",
            yaxis_title="Incident Count",
            plot_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
            paper_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
            font_color='#FFFFFF' if st.session_state["theme"] == "dark" else '#000000'
        )
        st.plotly_chart(fig, use_container_width=True)

#Help Expander
with st.expander("ℹ️ Help: How to Use the Railroad Incident Map"):
    st.markdown("""
    ### How to Use the Railroad Incident Map

    **Coloring the Map**  
    - "Color Map Markers By" in the sidebar controls whether markers (and scatter points) are colored by Accident_Type or Weather_Condition.

    **Select Mode**  
    - **Incident Details**: Markers represent individual incidents.  
      - Click a marker on the map to see details.  
      - "Radar Plot" in Incident Mode shows a single-incident radar.  
    - **State Details**: Markers represent states (just placeholders in this example).  
      - Click a state marker to see aggregate stats.  
      - "Radar Plot" in State Mode compares states.  

    **Visualization Mode**  
    - **Multi-Scatter Plots**:  
      - 2x2 grid of damage/fatalities/injuries vs. speed.  
      - Clicking a point re-centers the map on that incident.  
      - If in State Mode, you auto-switch to Incident Mode.  
    - **Radar Plot**:  
      - Compare states or see a single incident’s metrics in context of min–max.  
    - **Line Chart**:  
      - Uses integer `year` and `month` directly—no datetime conversion.  
      - Shows either total incidents or breaks out by weather condition.  
    - **Bar Chart**:  
      - Shows incident counts by state.  

    **Filters**  
    - Speed Category, Weather, Death Category, Injury Category, Damage Category  
    - **Start Year & End Year** for filtering by a range of years.

    **Clustering & Heatmap**  
    - Check these boxes for clustering or a heatmap overlay.  

    **Theme Toggle**  
    - Toggle between dark and light themes.

    **Clearing Incidents**  
    - The "Clear Incident Filter" button in Multi-Scatter resets any selected incident.
    """)

# Instructions to run the app
# streamlit run "C:\Users\yongj\OneDrive\Desktop\Visualization Project\dynamic_filters_map.py"
