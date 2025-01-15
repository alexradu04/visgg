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

# NEW: Import streamlit-plotly-events for click interactivity
from streamlit_plotly_events import plotly_events

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="Dynamic Railroad Incident Map")

################################################################################
# Minimal CSS injection for a dark background in main area & sidebar.
################################################################################
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

# Initialize theme toggle
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

def toggle_theme():
    st.session_state["theme"] = (
        "light" 
        if (st.session_state["theme"] == "dark" or st.session_state["theme"] == "states") 
        else "dark"
    )

# Add a button to toggle themes
if st.sidebar.button("Toggle Dark/Light Theme"):
    toggle_theme()

# Set map tiles based on the selected theme
map_tiles = {
    "dark": "CartoDB dark_matter",
    "light": "CartoDB positron",
    "states": "CartoDB positron"
}.get(st.session_state["theme"], "CartoDB dark_matter")

# Path to SQLite database
sqlite_db = r"C:\\Users\\yongj\\OneDrive\\Desktop\\Visualization Project\\railroad_incidents_cleanedMUT.db"

def get_filter_options(table_name, column_name):
    conn = sqlite3.connect(sqlite_db)
    query = f"SELECT DISTINCT {column_name} FROM {table_name};"
    options = [row[0] for row in conn.execute(query).fetchall()]
    conn.close()
    return options

# Sidebar filter options
speed_categories = get_filter_options("Train_Speed_Categories", "Speed_Category")
weather_conditions = get_filter_options("Weather_Conditions", "Weather_Condition")
year_groups = get_filter_options("Year_Groups", "Year_Group")
death_categories = get_filter_options("Death_Categories", "Death_Category")
injury_categories = get_filter_options("Injury_Categories", "Injury_Category")
damage_categories = get_filter_options("Equipment_Damage_Categories", "Damage_Category")

# Sidebar dropdown for visualization mode
st.sidebar.title("Visualization Mode")
visualization_mode = st.sidebar.selectbox(
    "Select Visualization", 
    ["Multi-Scatter Plots", "Radar Plot", "Line Chart"]
)

# Sidebar dropdown for selecting "Incident" or "State" detail mode
if "mode" not in st.session_state:
    st.session_state["mode"] = "Incident Details"

def switch_mode(new_mode):
    st.session_state["mode"] = new_mode

mode_options = ["Incident Details", "State Details"]
# We ensure the default session state value is used if none is set
selected_mode = st.sidebar.selectbox("Select Mode", mode_options, index=mode_options.index(st.session_state["mode"]))
if selected_mode != st.session_state["mode"]:
    switch_mode(selected_mode)

# Summary container in the sidebar
if st.session_state["mode"] == "State Details":
    st.sidebar.title("State Details")
    summary_container = st.sidebar.container()
else:
    st.sidebar.title("Incident Details")
    summary_container = st.sidebar.container()

################################################################################
# Two-column layout for filters
################################################################################
st.sidebar.title("Dynamic Filters")
col1, col2 = st.sidebar.columns(2)

with col1:
    selected_speed = st.selectbox("Speed Category", ["All"] + speed_categories)
    selected_weather = st.selectbox("Weather Condition", ["All"] + weather_conditions)
    selected_year = st.selectbox("Year Group", ["All"] + year_groups)

with col2:
    selected_death = st.selectbox("Death Category", ["All"] + death_categories)
    selected_injury = st.selectbox("Injury Category", ["All"] + injury_categories)
    selected_damage = st.selectbox("Damage Category", ["All"] + damage_categories)

enable_clustering = st.sidebar.checkbox("Enable Clustering", value=True)
show_heatmap = st.sidebar.checkbox("Show Heatmap")

# SQL query based on filters
query = """
SELECT R.latitude, R.longitud, R.trnspd, R.eqpdmg, R.trkdmg, R.caskld, R.casinj,
       C.Accident_Type, R.narr1, R.narr2, R.narr3, R.narr4, R.narr5, 
       R.narr6, R.narr7, R.narr8, R.narr9, R.narr10, 
       R.narr11, R.narr12, R.narr13, R.narr14, R.narr15, 
       R.state_name, W.Weather_Condition
FROM Railroad_Incidents R
LEFT JOIN Categorized_Incidents_By_ID C ON R.ID = C.ID
LEFT JOIN Train_Speed_Categories S ON R.ID = S.ID
LEFT JOIN Weather_Conditions W ON R.ID = W.ID
LEFT JOIN Year_Groups Y ON R.ID = Y.ID
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
if selected_year != "All":
    query += f" AND Y.Year_Group = '{selected_year}'"
if selected_death != "All":
    query += f" AND D.Death_Category = '{selected_death}'"
if selected_injury != "All":
    query += f" AND I.Injury_Category = '{selected_injury}'"
if selected_damage != "All":
    query += f" AND E.Damage_Category = '{selected_damage}'"

query += " LIMIT 2000;"  # Increase limit if you like

@st.cache_data
def do_db_query(query):
    conn = sqlite3.connect(sqlite_db)
    try:
        df = pd.read_sql_query(query, conn)
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

# Convert relevant columns to numeric
for col in ["trnspd", "eqpdmg", "trkdmg", "caskld", "casinj"]:
    filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

###############################################################################
# Let user choose color dimension
###############################################################################
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

###############################################################################
# Session states
###############################################################################
if "selected_incident" not in st.session_state:
    st.session_state["selected_incident"] = None

if "clicked_state" not in st.session_state:
    st.session_state["clicked_state"] = None

###############################################################################
# Create Folium map
###############################################################################
if st.session_state["selected_incident"] is not None:
    center_lat, center_lon = st.session_state["selected_incident"]
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=map_tiles)
else:
    m = folium.Map(location=[37.0902, -95.7129], zoom_start=5, tiles=map_tiles)

###############################################################################
# If in State Details mode, add an overlay label for the clicked state
###############################################################################
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

###############################################################################
# Add markers
###############################################################################
if st.session_state["mode"] == "State Details":
    # Example state markers
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
    # Incident Mode: markers for incidents
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

# Add heatmap if enabled
if show_heatmap:
    heat_data = filtered_data[["latitude", "longitud"]].dropna().values.tolist()
    if heat_data:
        HeatMap(heat_data).add_to(m)

###############################################################################
# Render the map and process clicks
###############################################################################
clicked_data = st_folium(m, use_container_width=True, height=600)

with summary_container:
    if clicked_data and clicked_data.get("last_clicked"):
        lat = clicked_data["last_clicked"]["lat"]
        lon = clicked_data["last_clicked"]["lng"]

        if st.session_state["mode"] == "State Details":
            # Find clicked state
            clickedState = find_state(lat, lon)
            st.session_state["clicked_state"] = str(clickedState)

            st.write(f"### Clicked State Coordinates: ({lat}, {lon})")
            st.write(f"### Clicked State: {clickedState}")

            wantedState = str(clickedState)
            state_data = filtered_data[filtered_data['state_name'] == wantedState]

            if state_data.empty:
                st.warning(f"No data available for state: {wantedState}")
            else:
                # Summaries
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
            # Incident Mode: Show that incident detail or do nothing here
            pass

    elif clicked_data and clicked_data.get("last_object_clicked") and st.session_state["mode"] == "Incident Details":
        lat = clicked_data["last_object_clicked"]["lat"]
        lon = clicked_data["last_object_clicked"]["lng"]
        # Find the matching incident
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
            st.session_state["selected_incident"] = (lat, lon)
        else:
            st.warning("No incident data found for that location.")

# 2x2 Scatter Plots (Equipment & Track vs Speed, then Killed & Injured vs Speed)
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

            # ------------------------------------------------------------------
            # NEW: Single-legend approach. We create separate traces per category
            # and place them in the correct subplot. This way we get a single
            # combined legend for all 4 scatterplots.
            # ------------------------------------------------------------------
            used_categories = set()
            unique_categories = chart_data[map_color_dimension].unique()

            for cat_val in unique_categories:
                subdf = chart_data[chart_data[map_color_dimension] == cat_val]
                cat_color = color_map.get(cat_val, "#808080")

                # 1) Equipment Damage vs Speed (top-left)
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
                    name=str(cat_val),              # Legend name
                    legendgroup=str(cat_val),       # Group so it appears as one legend item
                    showlegend=(cat_val not in used_categories)  # Only show once
                )
                fig.add_trace(eqp_trace, row=1, col=1)

                # 2) Track Damage vs Speed (top-right)
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

                # 3) Fatalities vs Speed (bottom-left)
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

                # 4) Injuries vs Speed (bottom-right)
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

                # Mark this category as used so only the first trace for each category
                # shows the legend item.
                used_categories.add(cat_val)

            # Update axes titles
            fig.update_xaxes(title_text="Train Speed (mph)", row=1, col=1)
            fig.update_yaxes(title_text="Equipment Damage ($)", row=1, col=1)
            fig.update_xaxes(title_text="Train Speed (mph)", row=1, col=2)
            fig.update_yaxes(title_text="Track Damage ($)", row=1, col=2)
            fig.update_xaxes(title_text="Train Speed (mph)", row=2, col=1)
            fig.update_yaxes(title_text="Killed", row=2, col=1)
            fig.update_xaxes(title_text="Train Speed (mph)", row=2, col=2)
            fig.update_yaxes(title_text="Injuries", row=2, col=2)

            # Make point-clicking more precise
            fig.update_layout(
                title_text=f"Damage, Fatalities & Injuries vs. Speed{scatter_title_suffix}",
                plot_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                paper_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                font_color='#FFFFFF' if st.session_state["theme"] == "dark" else '#000000',
                hovermode="closest",            # Ensures the nearest data point is picked
                clickmode="event+select",       # Ensures point selection is by the actual point
                legend=dict(
                    x=1.02,  # shift legend to the right
                    y=1.0,
                    xanchor="left",
                    yanchor="top",
                    title=f"{map_color_dimension}"
                )
            )

            st.markdown("**Click any point to zoom the map to that incident.**")

            # Use plotly_events to capture clicks
            selected_points = plotly_events(
                fig,
                click_event=True,
                hover_event=False,
                select_event=False,
                override_height=800,  # taller for 2x2 layout
                override_width="100%"
            )

            # (REMOVED the extra st.plotly_chart(...) to avoid duplication)

            # If the user clicked a point, we center the map and (if in State mode) swap to Incident mode
            if selected_points:
                point_index = selected_points[0]["pointIndex"]
                # Retrieve the subplot data from the *actual trace* they clicked
                trace_index = selected_points[0]["curveNumber"]
                # The clicked curve belongs to the trace among those we added.

                # Because we have multiple categories in multiple subplots,
                # we can figure out which subset (subdf) they clicked from "meta".
                # Plotly returns the same 'pointIndex' for the trace. Let's pull the lat/lon from fig.data[trace_index].meta

                # trace_index is the index in fig.data. We'll get the lat/lon from there.
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

elif visualization_mode == "Radar Plot":
    ############################
    # Radar Plot Implementation
    # with a Better Balanced Normalization (log + min–max)
    # and STATE-BASED interactivity
    ############################

    st.markdown("### Interactive Radar Plot (Balanced)")

    # 1. If in State Details mode and a state is clicked, filter to that state.
    if mode == "State Details" and clicked_data and clicked_data.get("last_clicked"):
        lat = clicked_data["last_clicked"]["lat"]
        lon = clicked_data["last_clicked"]["lng"]
        wantedState = str(find_state(lat, lon))
        radar_data = filtered_data[filtered_data['state_name'] == wantedState]
    else:
        # Otherwise, use all filtered data
        radar_data = filtered_data.copy()

    # Safety check for empty data
    if radar_data.empty:
        st.warning("No data available for the selected filters and/or clicked state to plot the Radar Chart.")
    else:
        # Columns of interest
        attrs = {
            "trnspd": "Average Speed (mph)",
            "trkdmg": "Track Damage ($)",
            "caskld": "Deaths",
            "casinj": "Injuries",
            "eqpdmg": "Equipment Damage ($)"
        }

        # This dict will hold the final single radar value (0–1) for each attribute.
        normalized_values = {}

        import numpy as np

        # 2. For each attribute, compute a "balanced" average on a log+min–max scale
        for col in attrs.keys():
            # Extract column data, drop NaNs
            col_data = radar_data[col].dropna()
            if col_data.empty:
                normalized_values[col] = 0.0
                continue

            # Shift so smallest value is ≥ 1 for log transform
            min_val_raw = col_data.min()
            shift_amount = 1 - min_val_raw if min_val_raw < 1 else 0
            col_data_shifted = col_data + shift_amount

            # Log transform
            col_data_log = np.log(col_data_shifted)

            # Min–max on the logged data
            col_log_min = col_data_log.min()
            col_log_max = col_data_log.max()
            if col_log_min == col_log_max:
                col_data_norm = [0.0] * len(col_data_log)
            else:
                col_data_norm = (col_data_log - col_log_min) / (col_log_max - col_log_min)

            # Final single value = mean of normalized distribution
            col_norm_mean = np.mean(col_data_norm)
            normalized_values[col] = col_norm_mean

        # 3. Prepare categories & single normalized value for each attribute
        radar_categories = list(attrs.values())
        radar_values = [normalized_values[col] for col in attrs.keys()]

        # 4. Create radar chart with Plotly's go.Scatterpolar
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_categories,
            fill='toself',
            fillcolor='rgba(0, 180, 255, 0.3)',
            line_color='rgba(0, 180, 255, 1)',
            marker=dict(symbol='circle', size=6, color='rgba(0, 180, 255, 1)'),
            hovertemplate="<b>%{theta}</b>: %{r:.2f}<extra></extra>"
        ))

        # 5. Style the Radar Plot (Dark Polygon Style)
        fig.update_layout(
            polar=dict(
                bgcolor='#1E1E1E',
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showline=True,
                    linewidth=2,
                    linecolor='rgba(255,255,255,0.2)',
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.2)',
                    gridwidth=1,
                    tickfont=dict(color='#FFFFFF'),
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
                ),
                angularaxis=dict(
                    visible=True,
                    showline=True,
                    linewidth=1,
                    linecolor='rgba(255,255,255,0.2)',
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.2)',
                    tickfont=dict(color='#FFFFFF'),
                    rotation=90,
                ),
            ),
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            font=dict(color='#FFFFFF'),
            showlegend=False,
            margin=dict(l=60, r=60, t=120, b=60),
        )

        # Title in range [0,1] for 'y'
        chart_title = "<b>Balanced Radar Plot</b>"
        if mode == "State Details" and not radar_data.empty:
            # Optionally show the state name in the title if you like
            state_name = radar_data['state_name'].iloc[0]
            chart_title += f"<br>{state_name}"

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

        # 6. Display the chart
        st.plotly_chart(fig, use_container_width=True)



elif visualization_mode == "Line Chart":
    st.markdown("### Line Chart - *Coming Soon!*")
    st.write("Line Chart functionality is not yet implemented.")

# Add Help Dropdown Menu
with st.expander("ℹ️ Help: How to Use the Railroad Incident Map"):
    st.markdown("""
    ### How to Use the Railroad Incident Map

    **Coloring the Map**  
    - "Color Map Markers By" in the sidebar controls whether markers (and scatter points) are colored by Accident_Type or Weather_Condition.

    **Select Mode**  
    - **Incident Details**: Markers represent individual incidents.  
      - Click a marker on the map to see details about that incident.  
    - **State Details**: Markers represent states.  
      - Click a state marker to see aggregate stats.  

    **Visualization Mode**  
    - **Multi-Scatter Plots**:  
      - Displays a 2x2 grid: 
        1) Equipment Damage vs Speed  
        2) Track Damage vs Speed  
        3) Fatalities vs Speed  
        4) Injuries vs Speed  
      - Click a point to "zoom" the Folium map and (if you’re in State Mode) automatically switch to Incident Mode.  

    **Dynamic Filters**  
    - Refine results by Speed Category, Weather, Year, Death, Injury, Damage categories.  

    **Clustering & Heatmap**  
    - Check these boxes for marker clustering or a heatmap overlay.

    **Theme Toggle**  
    - Switch between dark and light themes for better visual comfort.

    **Clearing Incidents**  
    - The "Clear Incident Filter" button resets the selected incident.
    """)


# Instructions to run the app
#streamlit run "C:\Users\yongj\OneDrive\Desktop\Visualization Project\dynamic_filters_map.py"
