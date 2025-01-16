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

st.set_page_config(layout="wide", page_title="Dynamic Railroad Incident Map")

# CSS for dark theme...
st.markdown("""
    <style>
    /* Make the main background dark */
    .main {
        background-color: #1E1E1E; 
    }
    /* Make the sidebar background a slightly different dark */
    section[data-testid="stSidebar"] {
        background-color: #2E2E2E; 
    }
    /* Change general text color to white */
    body, .markdown-text-container, .sidebar .sidebar-content {
        color: #FFFFFF;
    }
    /* Customize selectbox and other widgets */
    .css-1d391kg, .css-1aumxhk, .css-1b0td9e {
        background-color: #3E3E3E;
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)


# Toggle theme
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

# Choose tile based on theme
map_tiles = {
    "dark": "CartoDB dark_matter",
    "light": "CartoDB positron",
    "states": "CartoDB positron"
}.get(st.session_state["theme"], "CartoDB dark_matter")

# Path to your SQLite database
sqlite_db = r"C:\\Users\\yongj\\OneDrive\\Desktop\\Visualization Project\\railroad_incidents_cleanedMUT.db"

def get_filter_options(table_name, column_name):
    conn = sqlite3.connect(sqlite_db)
    query = f"SELECT DISTINCT {column_name} FROM {table_name};"
    options = [row[0] for row in conn.execute(query).fetchall()]
    conn.close()
    return options

# Get distinct categories
speed_categories = get_filter_options("Train_Speed_Categories", "Speed_Category")
weather_conditions = get_filter_options("Weather_Conditions", "Weather_Condition")
death_categories = get_filter_options("Death_Categories", "Death_Category")
injury_categories = get_filter_options("Injury_Categories", "Injury_Category")
damage_categories = get_filter_options("Equipment_Damage_Categories", "Damage_Category")

# Sidebar for visualization mode
st.sidebar.title("Visualization Mode")
visualization_mode = st.sidebar.selectbox(
    "Select Visualization", 
    ["Multi-Scatter Plots", "Radar Plot", "Line Chart", "Bar Chart"]
)

# Sidebar: "Incident" or "State" mode
if "mode" not in st.session_state:
    st.session_state["mode"] = "Incident Details"

def switch_mode(new_mode):
    st.session_state["mode"] = new_mode

mode_options = ["Incident Details", "State Details"]
selected_mode = st.sidebar.selectbox("Select Mode", mode_options, index=mode_options.index(st.session_state["mode"]))
if selected_mode != st.session_state["mode"]:
    switch_mode(selected_mode)

# Sidebar container for summary
if st.session_state["mode"] == "State Details":
    st.sidebar.title("State Details")
    summary_container = st.sidebar.container()
else:
    st.sidebar.title("Incident Details")
    summary_container = st.sidebar.container()

# Two columns for filters
st.sidebar.title("Dynamic Filters")
col1, col2 = st.sidebar.columns(2)

with col1:
    selected_speed = st.selectbox("Speed Category", ["All"] + speed_categories)
    selected_weather = st.selectbox("Weather Condition", ["All"] + weather_conditions)
    selected_year = st.text_input("Filter by Year (e.g. 2015, or blank for all):", "")

with col2:
    selected_death = st.selectbox("Death Category", ["All"] + death_categories)
    selected_injury = st.selectbox("Injury Category", ["All"] + injury_categories)
    selected_damage = st.selectbox("Damage Category", ["All"] + damage_categories)

enable_clustering = st.sidebar.checkbox("Enable Clustering", value=True)
show_heatmap = st.sidebar.checkbox("Show Heatmap")

# Build the SQL query
query = """
SELECT 
    R.latitude,
    R.longitud,
    R.trnspd,
    R.eqpdmg,
    R.trkdmg,
    R.caskld,
    R.casinj,
    R.year,         /* <-- Make sure this matches the exact column name in your DB */
    R.month,        /* <-- Same note for month */
    C.Accident_Type,
    R.narr1, R.narr2, R.narr3, R.narr4, R.narr5, 
    R.narr6, R.narr7, R.narr8, R.narr9, R.narr10, 
    R.narr11, R.narr12, R.narr13, R.narr14, R.narr15, 
    R.state_name,
    W.Weather_Condition
FROM Railroad_Incidents R
LEFT JOIN Categorized_Incidents_By_ID C ON R.ID = C.ID
LEFT JOIN Train_Speed_Categories S ON R.ID = S.ID
LEFT JOIN Weather_Conditions W ON R.ID = W.ID
LEFT JOIN Death_Categories D ON R.ID = D.ID
LEFT JOIN Injury_Categories I ON R.ID = I.ID
LEFT JOIN Equipment_Damage_Categories E ON R.ID = E.ID
WHERE 1=1
"""

# Apply filters
if selected_speed != "All":
    query += f" AND S.Speed_Category = '{selected_speed}'"
if selected_weather != "All":
    query += f" AND W.Weather_Condition = '{selected_weather}'"
if selected_year.strip():
    # Filter by a single year
    # If your "year" column is INT in the DB, numeric comparison is fine:
    query += f" AND R.year = {selected_year.strip()}"

if selected_death != "All":
    query += f" AND D.Death_Category = '{selected_death}'"
if selected_injury != "All":
    query += f" AND I.Injury_Category = '{selected_injury}'"
if selected_damage != "All":
    query += f" AND E.Damage_Category = '{selected_damage}'"

# Limit the result set to 1000 for performance
query += " LIMIT 1000;"

@st.cache_data
def do_db_query(sql_query):
    conn = sqlite3.connect(sqlite_db)
    try:
        df = pd.read_sql_query(sql_query, conn)
    except sqlite3.OperationalError as e:
        st.error(f"Error executing query: {e}")
        conn.close()
        st.stop()
    conn.close()
    return df

filtered_data = do_db_query(query)

# Combine narrative columns
def combine_narratives(row):
    narratives = [row[f"narr{i}"] for i in range(1, 16) if pd.notnull(row[f"narr{i}"])]
    return " ".join(narratives)

filtered_data["description"] = filtered_data.apply(combine_narratives, axis=1)

# Convert columns to numeric (if relevant)
for col in ["trnspd", "eqpdmg", "trkdmg", "caskld", "casinj", "year", "month"]:
    filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

# Let user choose color dimension
map_color_dimension = st.sidebar.selectbox(
    "Color Map Markers By", 
    ["Accident_Type", "Weather_Condition"], 
    index=0
)

def build_unified_color_map(df, color_dimension):
    unique_vals = df[color_dimension].dropna().unique()
    color_palette = px.colors.qualitative.Plotly
    color_map = {
        val: color_palette[i % len(color_palette)]
        for i, val in enumerate(unique_vals)
    }
    return color_map

color_map = build_unified_color_map(filtered_data, map_color_dimension)

# Session state for selected incidents/states
if "selected_incident" not in st.session_state:
    st.session_state["selected_incident"] = None
if "clicked_state" not in st.session_state:
    st.session_state["clicked_state"] = None

# Build map
def build_map_center():
    if st.session_state["selected_incident"] is not None:
        center_lat, center_lon = st.session_state["selected_incident"]
        return [center_lat, center_lon], 13
    else:
        return [37.0902, -95.7129], 5

map_center, map_zoom = build_map_center()
m = folium.Map(location=map_center, zoom_start=map_zoom, tiles=map_tiles)

# If in State Details, optionally show a label for the clicked state
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

# Add markers
if st.session_state["mode"] == "State Details":
    # Simple placeholders for states
    if enable_clustering:
        marker_cluster = MarkerCluster().add_to(m)
    else:
        marker_cluster = m

    state_markers = [
        {"lat": 34.0489, "lon": -111.0937, "state": "Arizona"},
        {"lat": 40.7128, "lon": -74.0060, "state": "New York"},
    ]
    for state_info in state_markers:
        folium.Marker(
            location=[state_info["lat"], state_info["lon"]],
            tooltip=f"State: {state_info['state']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

else:
    # Incident mode => plot actual incidents
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

# Heatmap
if show_heatmap:
    heat_data = filtered_data[["latitude", "longitud"]].dropna().values.tolist()
    if heat_data:
        HeatMap(heat_data).add_to(m)

clicked_data = st_folium(m, use_container_width=True, height=600)

# Sidebar details after click
with summary_container:
    if clicked_data and clicked_data.get("last_clicked"):
        lat = clicked_data["last_clicked"]["lat"]
        lon = clicked_data["last_clicked"]["lng"]

        if st.session_state["mode"] == "State Details":
            # Identify which state was clicked
            clickedState = find_state(lat, lon)
            st.session_state["clicked_state"] = str(clickedState)

            st.write(f"### Clicked State Coordinates: ({lat}, {lon})")
            st.write(f"### Clicked State: {clickedState}")

            wantedState = str(clickedState)
            state_data = filtered_data[filtered_data['state_name'] == wantedState]

            if state_data.empty:
                st.warning(f"No data available for state: {wantedState}")
            else:
                total_incidents = state_data.shape[0]
                st.write(f"**Total incidents in {wantedState}:** {total_incidents}")

                if 'trnspd' in state_data.columns and state_data['trnspd'].notnull().any():
                    avg_speed = state_data['trnspd'].mean()
                    st.write(f"**Average train speed in {wantedState}:** {avg_speed:.2f} mph")
                else:
                    st.write("**Average train speed:** Data not available.")

                total_fatalities = state_data['caskld'].sum()
                total_injuries = state_data['casinj'].sum()
                st.write(f"**Total fatalities in {wantedState}:** {int(total_fatalities)}")
                st.write(f"**Total injuries in {wantedState}:** {int(total_injuries)}")

                if 'Weather_Condition' in state_data.columns and state_data['Weather_Condition'].notnull().any():
                    most_common_weather = (
                        state_data['Weather_Condition']
                        .value_counts()
                        .idxmax()
                    )
                    st.write(f"**Most common weather condition in {wantedState}:** {most_common_weather}")
                else:
                    st.write("**Most common weather condition:** Data not available.")

                incident_breakdown = state_data['Accident_Type'].value_counts()
                st.write(f"**Incident types breakdown in {wantedState}:**")
                st.write(incident_breakdown)

        else:
            # Incident mode
            st.session_state["selected_incident"] = (lat, lon)
            incident = filtered_data[
                (filtered_data["latitude"] == lat) & (filtered_data["longitud"] == lon)
            ]
            if not incident.empty:
                row = incident.iloc[0]
                st.write("### Incident Details")
                st.write(f"**Accident Type:** {row['Accident_Type']}")
                st.write(f"**Injuries:** {row['casinj']}")
                st.write(f"**Deaths:** {row['caskld']}")
                st.write(f"**Description:** {row['description']}")
                st.write(f"**State:** {row['state_name']}")
                st.write(f"**Train Speed:** {row['trnspd']} mph")
                st.write(f"**Equipment Damage:** ${row['eqpdmg']}")
                st.write(f"**Track Damage:** ${row['trkdmg']}")
                st.write(f"**Weather Condition:** {row['Weather_Condition']}")
            else:
                st.warning("No incident data found for that location.")

    elif clicked_data and clicked_data.get("last_object_clicked") and st.session_state["mode"] == "Incident Details":
        lat = clicked_data["last_object_clicked"]["lat"]
        lon = clicked_data["last_object_clicked"]["lng"]
        st.session_state["selected_incident"] = (lat, lon)

        incident = filtered_data[
            (filtered_data["latitude"] == lat) & (filtered_data["longitud"] == lon)
        ]
        if not incident.empty:
            row = incident.iloc[0]
            st.write("### Incident Details")
            st.write(f"**Accident Type:** {row['Accident_Type']}")
            st.write(f"**Injuries:** {row['casinj']}")
            st.write(f"**Deaths:** {row['caskld']}")
            st.write(f"**Description:** {row['description']}")
            st.write(f"**State:** {row['state_name']}")
            st.write(f"**Train Speed:** {row['trnspd']} mph")
            st.write(f"**Equipment Damage:** ${row['eqpdmg']}")
            st.write(f"**Track Damage:** ${row['trkdmg']}")
            st.write(f"**Weather Condition:** {row['Weather_Condition']}")
        else:
            st.warning("No incident data found for that location.")

# Visualization Modes

# 1) Multi-Scatter Plots
if visualization_mode == "Multi-Scatter Plots":
    st.markdown("## Multi-Scatter Plots: Damage, Fatalities, Injuries vs. Speed")

    # If in state mode and a state is selected, limit data to that state
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
            # Build 2x2 subplots
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

            used_categories = set()
            unique_categories = chart_data[map_color_dimension].unique()

            for cat_val in unique_categories:
                subdf = chart_data[chart_data[map_color_dimension] == cat_val]
                cat_color = color_map.get(cat_val, "#808080")

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
                    showlegend=(cat_val not in used_categories)
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
                    showlegend=(cat_val not in used_categories)
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
                    showlegend=(cat_val not in used_categories)
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
                    showlegend=(cat_val not in used_categories)
                )
                fig.add_trace(inj_trace, row=2, col=2)

                used_categories.add(cat_val)

            # Highlight the selected incident if any
            if st.session_state["selected_incident"] is not None:
                hilat, hilon = st.session_state["selected_incident"]
                # find that row in chart_data
                row_match = chart_data[
                    (chart_data["latitude"] == hilat) & (chart_data["longitud"] == hilon)
                ]
                if not row_match.empty:
                    # We'll highlight it in all 4 subplots with star markers
                    r = row_match.iloc[0]
                    highlight_x_eqp = r["trnspd"]
                    highlight_y_eqp = r["eqpdmg"]
                    highlight_x_trk = r["trnspd"]
                    highlight_y_trk = r["trkdmg"]
                    highlight_x_fat = r["trnspd"]
                    highlight_y_fat = r["caskld"]
                    highlight_x_inj = r["trnspd"]
                    highlight_y_inj = r["casinj"]

                    # Add four single-point highlight traces
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

            # Update axes titles
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
                    x=1.02,
                    y=1.0,
                    xanchor="left",
                    yanchor="top",
                    title=f"{map_color_dimension}"
                )
            )

            st.markdown("**Click any point to zoom the map to that incident.**")

            # plotly_events to capture scatter clicks
            selected_points = plotly_events(
                fig,
                click_event=True,
                hover_event=False,
                select_event=False,
                override_height=800,
                override_width="100%"
            )

            # If the user clicked a point on the scatter, center the map there
            if selected_points:
                point_index = selected_points[0]["pointIndex"]
                trace_index = selected_points[0]["curveNumber"]
                lat_val = float(fig.data[trace_index].meta[point_index][0])
                lon_val = float(fig.data[trace_index].meta[point_index][1])
                st.session_state["selected_incident"] = (lat_val, lon_val)
                # If currently in State mode, switch to Incident mode
                if st.session_state["mode"] == "State Details":
                    st.session_state["mode"] = "Incident Details"
                st.info(f"Switched to 'Incident Details' mode and centered map at: (lat={lat_val}, lon={lon_val}).")

            # Button to clear incident filter
            if st.button("Clear Incident Filter"):
                st.session_state["selected_incident"] = None
                st.info("Incident filter cleared. Map is reset to the default view.")

# 2) Radar Plot
elif visualization_mode == "Radar Plot":
    st.markdown("## Radar Plot")

    # 2A. State-based radar (comparison) if in State Mode
    if st.session_state["mode"] == "State Details":
        st.markdown("#### State Radar Plot (Comparison)")
        if st.session_state["clicked_state"]:
            base_state = st.session_state["clicked_state"]
            base_data = filtered_data[filtered_data['state_name'] == base_state]

            #dropdown to pick a comparison state
            all_states = sorted(filtered_data['state_name'].dropna().unique())
            comparison_state = st.sidebar.selectbox(
                "Compare with another State",
                ["None"] + all_states,
                index=0
            )

            import numpy as np

            def compute_radar_values_state(state_df):
                """Compute normalized (0-1) radar values for the given state's distribution using log+min–max transform."""
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

                    # SHIFT + LOG + MIN–MAX
                    min_val_raw = col_data.min()
                    shift_amount = 1 - min_val_raw if min_val_raw < 1 else 0
                    col_data_shifted = col_data + shift_amount

                    col_data_log = np.log(col_data_shifted)
                    col_log_min, col_log_max = col_data_log.min(), col_data_log.max()

                    if col_log_min == col_log_max:
                        col_data_norm = 0.0
                    else:
                        col_data_norm = (col_data_log - col_log_min) / (col_log_max - col_log_min)

                    # final single value = mean
                    if isinstance(col_data_norm, float):
                        col_norm_mean = col_data_norm
                    else:
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
                # Add base state
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

                # If user chose a comparison state
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
                        x=0.5,
                        y=0.95,
                        xanchor='center',
                        yanchor='top',
                        font=dict(size=18)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Click on a state marker in the map to see its Radar Plot.")

    # 2B. Incident-based radar if in Incident Mode & an incident is selected
    if st.session_state["mode"] == "Incident Details":
        st.markdown("#### Incident Radar Plot (Single Incident)")
        if st.session_state["selected_incident"] is not None:
            lat_inc, lon_inc = st.session_state["selected_incident"]
            inc_row = filtered_data[
                (filtered_data["latitude"] == lat_inc) & (filtered_data["longitud"] == lon_inc)
            ]
            if not inc_row.empty:
                # We'll do a single-incident radar.
                # We'll scale by min-max across the entire filtered_data, so the user sees how big or small
                # this incident is relative to the entire dataset.
                rowvals = inc_row.iloc[0]

                # We define the same 5 attributes
                attributes = ["trnspd", "eqpdmg", "trkdmg", "caskld", "casinj"]
                display_labels = [
                    "Train Speed",
                    "Equip Damage",
                    "Track Damage",
                    "Deaths",
                    "Injuries"
                ]

                # Compute min & max from entire filtered_data
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

                # Now build normalized [0..1] value for this single incident
                incident_radar_vals = []
                for col in attributes:
                    val = rowvals[col] if not pd.isnull(rowvals[col]) else 0.0
                    rng = maxs[col] - mins[col]
                    if rng == 0:
                        norm_val = 0.0
                    else:
                        norm_val = (val - mins[col]) / rng
                    incident_radar_vals.append(norm_val)

                # Plot a single polar
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

# 3) Line Chart (Not working yet)
elif visualization_mode == "Line Chart":
    st.markdown("### Line Chart: Incidents Over Time (Year-Month)")

    # We'll group by year-month from the columns "year" and "month"
    line_data = filtered_data.dropna(subset=["year", "month"])
    if line_data.empty:
        st.warning("No data available for the selected filters or missing year/month.")
    else:
        # Build a date column from year-month (arbitrary day=1)
        line_data["date"] = pd.to_datetime(
            line_data.apply(lambda r: f"{int(r.year)}-{int(r.month)}-01", axis=1),
            format="%Y-%m-%d",
            errors="coerce"
        )
        line_data = line_data.dropna(subset=["date"])
        if line_data.empty:
            st.warning("No valid date could be formed from year/month.")
        else:
            # Let user pick if we want "Total Incidents" or "By Weather Condition"
            line_category = st.radio("Line Category", ["Total Incidents", "By Weather Condition"], index=0)

            if line_category == "Total Incidents":
                # Group by date => count
                grouped = line_data.groupby("date").size().reset_index(name="count")
                fig = px.line(
                    grouped,
                    x="date",
                    y="count",
                    title="Total Incidents by Year-Month",
                    markers=True
                )
                fig.update_traces(line_color="cyan")
            else:
                # By Weather => group by date+weather => multiple lines
                line_data["Weather_Condition"] = line_data["Weather_Condition"].fillna("Unknown")
                grouped = line_data.groupby(["date", "Weather_Condition"]).size().reset_index(name="count")
                fig = px.line(
                    grouped,
                    x="date",
                    y="count",
                    color="Weather_Condition",
                    title="Incidents by Year-Month & Weather Condition",
                    markers=True
                )

            fig.update_layout(
                plot_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                paper_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                font_color='#FFFFFF' if st.session_state["theme"] == "dark" else '#000000'
            )
            st.plotly_chart(fig, use_container_width=True)


# 4) Bar Chart
elif visualization_mode == "Bar Chart":
    st.markdown("### Bar Chart: Incidents by State")

    # Group by state => count
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

# Add Help Dropdown Menu
with st.expander("ℹ️ Help: How to Use the Railroad Incident Map"):
    st.markdown("""
    ### How to Use the Railroad Incident Map

    **Coloring the Map**  
    - "Color Map Markers By" in the sidebar controls whether markers (and scatter points) are colored by Accident_Type or Weather_Condition.

    **Select Mode**  
    - **Incident Details**: Markers represent individual incidents.  
      - Click a marker on the map to see details about that incident.  
      - You also get an **Incident Radar Plot** in "Radar Plot" visualization.  
    - **State Details**: Markers represent states.  
      - Click a state marker to see aggregate stats.  
      - You get a **State Radar Plot** in "Radar Plot" visualization, with optional comparison.  

    **Visualization Mode**  
    - **Multi-Scatter Plots**:  
      - 2x2 grid of damage, fatalities, injuries vs. train speed.  
      - Clicking a point will recenter the map on that incident.  
      - If in State Mode, you'll automatically switch to Incident Mode.  
    - **Radar Plot**:  
      - In **State Mode**: Compare two states.  
      - In **Incident Mode**: See a single incident’s metrics vs. min–max of all filtered incidents.  
    - **Line Chart**:  
      - Uses `incident_year` and `incident_month` to show incident counts over time.  
      - Choose "Total Incidents" or "By Weather Condition" lines.  
    - **Bar Chart**:  
      - Shows incident counts by state.  

    **Dynamic Filters**  
    - Speed Category, Weather, Year, Death, Injury, Damage, etc.  

    **Clustering & Heatmap**  
    - Check these boxes for clustering or a heatmap overlay.

    **Theme Toggle**  
    - Toggle between dark and light themes.

    **Clearing Incidents**  
    - The "Clear Incident Filter" button in the Multi-Scatter Plot resets any selected incident.
    """)

# Instructions to run the app
# streamlit run "C:\Users\yongj\OneDrive\Desktop\Visualization Project\dynamic_filters_map.py"
