import os
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import sqlite3
from states_json_util import find_state
import plotly.express as px

# Set Streamlit page config (layout="wide" helps the map fill more space)
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

# Loading filter options
speed_categories = get_filter_options("Train_Speed_Categories", "Speed_Category")
weather_conditions = get_filter_options("Weather_Conditions", "Weather_Condition")
year_groups = get_filter_options("Year_Groups", "Year_Group")
death_categories = get_filter_options("Death_Categories", "Death_Category")
injury_categories = get_filter_options("Injury_Categories", "Injury_Category")
damage_categories = get_filter_options("Equipment_Damage_Categories", "Damage_Category")

# Sidebar dropdown for selecting visualization mode
st.sidebar.title("Visualization Mode")
visualization_mode = st.sidebar.selectbox("Select Visualization", ["Dynamic Bar Chart", "Radar Plot", "Line Chart"])

# Sidebar dropdown for selecting mode
mode = st.sidebar.selectbox("Select Mode", ["Incident Details", "State Details"])

# Depending on mode, set sidebar title and create a summary container
if mode == "State Details":
    st.sidebar.title("State Details")
    summary_container = st.sidebar.container()
else:
    st.sidebar.title("Incident Details")
    summary_container = st.sidebar.container()  # Container created for Incident Details

################################################################################
# Two-column layout for filters in the sidebar.
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

# SQL query for filters
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

# Adding filters
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

query += " LIMIT 1000;"

@st.cache_data
def do_db_query(query):
    conn = sqlite3.connect(sqlite_db)
    try:
        filtered_data = pd.read_sql_query(query, conn)
    except sqlite3.OperationalError as e:
        st.error(f"Error executing query: {e}")
        conn.close()
        st.stop()
    conn.close()
    return filtered_data

filtered_data = do_db_query(query)

# Combine narrative columns
def combine_narratives(row):
    narratives = [row[f"narr{i}"] for i in range(1, 16) if pd.notnull(row[f"narr{i}"])]
    return " ".join(narratives)

filtered_data["description"] = filtered_data.apply(combine_narratives, axis=1)

# Convert relevant columns to numeric, coercing errors to NaN
numeric_columns = ['trnspd', 'eqpdmg', 'trkdmg', 'caskld', 'casinj']
for col in numeric_columns:
    filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

# Assign colors to accident types (improved palette)
unique_accident_types = filtered_data["Accident_Type"].unique()
color_palette = px.colors.qualitative.Plotly  # Using Plotly's qualitative palette
accident_colors = {
    accident_type: color_palette[i % len(color_palette)]
    for i, accident_type in enumerate(unique_accident_types)
}

# Sidebar for displaying selected marker details
# incident_details_placeholder = st.sidebar.empty()  # Removed

# Create the map
m = folium.Map(location=[37.0902, -95.7129], zoom_start=5, tiles=map_tiles)

if mode == "State Details":
    # State Mode: Add state markers
    state_markers = [
        {"lat": 34.0489, "lon": -111.0937, "state": "Arizona"},
        {"lat": 40.7128, "lon": -74.0060, "state": "New York"},
        # Add more states as needed
    ]
    for state in state_markers:
        folium.Marker(
            location=[state["lat"], state["lon"]],
            tooltip=f"State: {state['state']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
else:
    # Incident Mode: Add incident markers and clustering
    if enable_clustering:
        marker_cluster = MarkerCluster().add_to(m)

    marker_data = []
    for _, row in filtered_data.iterrows():
        accident_type = row["Accident_Type"]
        color = accident_colors.get(accident_type, "gray")
        description = row["description"]

        # Skip rows with missing latitude or longitude
        if pd.isnull(row["latitude"]) or pd.isnull(row["longitud"]):
            continue

        # **Remove popup from markers**
        # Previously, popups were added here. Now, we omit them.
        # popup_content = f"""
        # <b>Accident Type:</b> {accident_type}<br>
        # <b>Injuries:</b> {row['casinj']}<br>
        # <b>Deaths:</b> {row['caskld']}<br>
        # <b>Description:</b> {description}
        # """

        marker = folium.Marker(
            location=[row["latitude"], row["longitud"]],
            icon=folium.Icon(color=color),
            tooltip="Click for details"
            # popup=folium.Popup(popup_content, max_width=300)  # Removed
        )

        if enable_clustering:
            marker_cluster.add_child(marker)
        else:
            marker.add_to(m)

        # Store marker data for sidebar display
        marker_data.append({
            "lat": row["latitude"],
            "lon": row["longitud"],
            "accident_type": accident_type,
            "description": description,
            "injuries": row["casinj"],
            "deaths": row["caskld"],
            "state_name": row["state_name"],
            "trnspd": row.get("trnspd", None),
            "eqpdmg": row.get("eqpdmg", None),
            "trkdmg": row.get("trkdmg", None),
            "caskld": row.get("caskld", None),
            "casinj": row.get("casinj", None),
            "Weather_Condition": row.get("Weather_Condition", "Unknown")
        })

    # Add heatmap if enabled
    if show_heatmap:
        heat_data = filtered_data[["latitude", "longitud"]].dropna().values.tolist()
        HeatMap(heat_data).add_to(m)

###############################################################################
# Make the map fill the available screen width and increase its height.
###############################################################################
clicked_data = st_folium(m, use_container_width=True, height=600)

with summary_container:
    if clicked_data and clicked_data.get("last_clicked"):
        lat = clicked_data["last_clicked"]["lat"]
        lon = clicked_data["last_clicked"]["lng"]

        if mode == "State Details":
            # In State Mode: Show details for clicked state
            clickedState = find_state(lat, lon)
            st.write(f"### Clicked State Coordinates: ({lat}, {lon})")
            st.write(f"### Clicked State: {clickedState}")

            wantedState = str(clickedState) 
            state_data = filtered_data[filtered_data['state_name'] == wantedState]

            if state_data.empty:
                st.warning(f"No data available for state: {wantedState}")
            else:
                # 1. Total Number of Incidents
                total_incidents = state_data.shape[0]
                st.write(f"**Total incidents in {wantedState}:** {total_incidents}")

                # 2. Average Train Speed During Incidents
                if 'trnspd' in state_data.columns and state_data['trnspd'].notnull().any():
                    avg_speed = state_data['trnspd'].mean()
                    st.write(f"**Average train speed in {wantedState}:** {avg_speed:.2f} mph")
                else:
                    st.write("**Average train speed:** Data not available.")

                # 3. Number of Fatalities and Injuries
                total_fatalities = state_data['caskld'].sum()
                total_injuries = state_data['casinj'].sum()
                st.write(f"**Total fatalities in {wantedState}:** {int(total_fatalities)}")
                st.write(f"**Total injuries in {wantedState}:** {int(total_injuries)}")

                # 4. Most Common Weather Condition
                if 'Weather_Condition' in state_data.columns and state_data['Weather_Condition'].notnull().any():
                    most_common_weather = (
                        state_data['Weather_Condition']
                        .value_counts()
                        .idxmax()
                    )
                    st.write(f"**Most common weather condition in {wantedState}:** {most_common_weather}")
                else:
                    st.write("**Most common weather condition:** Data not available.")

                # 5. Incident Types Breakdown
                incident_breakdown = state_data['Accident_Type'].value_counts()
                st.write(f"**Incident types breakdown in {wantedState}:**")
                st.write(incident_breakdown)

    elif clicked_data and clicked_data.get("last_object_clicked") and mode == "Incident Details":
        # In Incident Mode: Show details for clicked marker
        lat = clicked_data["last_object_clicked"]["lat"]
        lon = clicked_data["last_object_clicked"]["lng"]
        # Find the incident matching the clicked lat and lon
        incident = next((item for item in marker_data if item["lat"] == lat and item["lon"] == lon), None)
        if incident:
            st.write(f"### Incident Details")
            st.write(f"**Accident Type:** {incident['accident_type']}")
            st.write(f"**Injuries:** {incident['injuries']}")
            st.write(f"**Deaths:** {incident['deaths']}")
            st.write(f"**Description:** {incident['description']}")
            st.write(f"**State:** {incident['state_name']}")
            st.write(f"**Train Speed:** {incident['trnspd']} mph")
            st.write(f"**Equipment Damage:** ${incident['eqpdmg']}")
            st.write(f"**Track Damage:** ${incident['trkdmg']}")
            st.write(f"**Weather Condition:** {incident['Weather_Condition']}")
        else:
            st.warning("No incident data found for the clicked location.")

###############################################################################
# Dynamic Bar Chart Visualization
###############################################################################
if visualization_mode == "Dynamic Bar Chart":
    st.markdown("### Dynamic Railroad Incidents Bar Chart")
    
    # Determine if a state is selected
    if mode == "State Details" and clicked_data and clicked_data.get("last_clicked"):
        wantedState = str(find_state(lat, lon))
        chart_data = filtered_data[filtered_data['state_name'] == wantedState]
    else:
        chart_data = filtered_data.copy()
    
    if chart_data.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Dropdowns for x-axis and y-axis
        x_axis_options = {
            "Train Speed (mph)": "trnspd",
        }
        
        y_axis_options = {
            "Equipment Damage ($)": "eqpdmg",
            "Track Damage ($)": "trkdmg",
            "Total Killed": "caskld",
            "Total Injured": "casinj"
        }
        
        # Using Streamlit's container for layout
        bar_chart_container = st.container()
        
        with bar_chart_container:
            bar_col1, bar_col2 = st.columns(2)
            with bar_col1:
                x_axis = st.selectbox("Select X-axis", options=list(x_axis_options.keys()), key="bar_x_axis")
            with bar_col2:
                y_axis = st.selectbox("Select Y-axis", options=list(y_axis_options.keys()), key="bar_y_axis")
            
            # Option for stacked bar chart by weather condition
            stack_by_weather = st.checkbox("Stack by Weather Condition", key="stack_weather")
            
            # Filtered data for selected columns
            selected_columns = [x_axis_options[x_axis], y_axis_options[y_axis], "Weather_Condition"]
            bar_filtered_data = chart_data[selected_columns].dropna()
            
            if bar_filtered_data.empty:
                st.warning("No data available for the selected axes.")
            else:
                # Rename columns for readability
                bar_filtered_data.columns = ["X", "Y", "Weather"]
                
                # Create the bar chart
                if stack_by_weather:
                    fig = px.bar(
                        bar_filtered_data,
                        x="X",
                        y="Y",
                        color="Weather",
                        barmode="stack",
                        labels={"X": x_axis, "Y": y_axis, "Weather": "Weather Condition"},
                        title=f"{y_axis} vs {x_axis} (Stacked by Weather Condition)",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                else:
                    weather_unique = sorted(bar_filtered_data["Weather"].unique())
                    weather_filter = st.selectbox("Select Weather Condition", ["All"] + weather_unique, key="bar_weather_filter")
                    if weather_filter != "All":
                        bar_filtered_data = bar_filtered_data[bar_filtered_data["Weather"] == weather_filter]
                        title_suffix = f" for Weather: {weather_filter}"
                    else:
                        title_suffix = ""
                    fig = px.bar(
                        bar_filtered_data,
                        x="X",
                        y="Y",
                        labels={"X": x_axis, "Y": y_axis},
                        title=f"{y_axis} vs {x_axis}{title_suffix}"
                    )
                
                # Adjust layout based on theme
                fig.update_layout(
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    plot_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                    paper_bgcolor='#1E1E1E' if st.session_state["theme"] == "dark" else '#FFFFFF',
                    font_color='#FFFFFF' if st.session_state["theme"] == "dark" else '#000000'
                )
                
                st.plotly_chart(fig, use_container_width=True)

elif visualization_mode == "Radar Plot":
    st.markdown("### Radar Plot - *Coming Soon!*")
    st.write("Radar Plot functionality is not yet implemented.")

elif visualization_mode == "Line Chart":
    st.markdown("### Line Chart - *Coming Soon!*")
    st.write("Line Chart functionality is not yet implemented.")

################################################################################
# Add Help Dropdown Menu
################################################################################
with st.expander("ℹ️ Help: How to Use the Railroad Incident Map"):
    st.markdown("""
    ### How to Use the Railroad Incident Map

    - **Visualization Mode:** Choose the type of visualization you want to see. Currently, only the **Dynamic Bar Chart** is available.
    
    - **Select Mode:** 
        - **Incident Details:** View individual incidents on the map. Click on markers to see detailed information in the sidebar.
        - **State Details:** View aggregated data for specific states. Click on state markers to see detailed statistics in the sidebar.
    
    - **Dynamic Filters:** Use the filters to narrow down the incidents based on speed, weather, year, death, injury, and damage categories.
    
    - **Enable Clustering:** Toggle clustering to group nearby incident markers for better map readability.
    
    - **Show Heatmap:** Display a heatmap overlay to visualize areas with high incident concentrations.
    
    - **Dynamic Bar Chart:** When selected, customize the bar chart by choosing different X and Y axes and optionally stacking by weather conditions.
    
    - **Theme Toggle:** Switch between dark and light themes for better visual comfort.
    
    ### Interactions:
    - **Map Click:**
        - **Incident Details Mode:** Click on an incident marker to view its details in the sidebar under "Incident Details".
        - **State Details Mode:** Click on a state marker to view its details in the sidebar under "State Details".
    
    - **Sidebar Controls:** All filters and visualization options are available in the sidebar for easy access.
    
    ### Future Enhancements:
    - **Radar Plot:** Compare multiple variables in a single visualization.
    - **Line Chart:** Track trends over time.
    """)

################################################################################
#Instructions to run the app
################################################################################
#To run the app, use the following command in your terminal:
#streamlit run "C:\Users\yongj\OneDrive\Desktop\Visualization Project\dynamic_filters_map.py"
