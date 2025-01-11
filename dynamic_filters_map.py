import os
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import sqlite3
from states_json_util import find_state

#Set Streamlit page config (layout="wide" helps the map fill more space)
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
    body, .markdown-text-container {
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

#Initialize theme toggle
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

def toggle_theme():
    st.session_state["theme"] = (
        "light" 
        if (st.session_state["theme"] == "dark" or st.session_state["theme"] == "states") 
        else "dark"
    )

#Add a button to toggle themes
if st.sidebar.button("Toggle Dark/Light Theme"):
    toggle_theme()

#Set map tiles based on the selected theme
match st.session_state["theme"]:
    case "dark":
        map_tiles = "CartoDB dark_matter"
    case "light":
        map_tiles = "CartoDB positron"
    case "states":
        map_tiles = "CartoDB positron"
    case _:
        map_tiles = "CartoDB dark_matter"

#Path to SQLite database
sqlite_db = "C://Users//yongj//OneDrive//Desktop//Visualization Project//railroad_incidents_cleanedMUT.db"

conn = sqlite3.connect(sqlite_db)

def get_filter_options(table_name, column_name):
    query = f"SELECT DISTINCT {column_name} FROM {table_name};"
    return [row[0] for row in conn.execute(query).fetchall()]

#Loading filter options
speed_categories = get_filter_options("Train_Speed_Categories", "Speed_Category")
weather_conditions = get_filter_options("Weather_Conditions", "Weather_Condition")
year_groups = get_filter_options("Year_Groups", "Year_Group")
death_categories = get_filter_options("Death_Categories", "Death_Category")
injury_categories = get_filter_options("Injury_Categories", "Injury_Category")
damage_categories = get_filter_options("Equipment_Damage_Categories", "Damage_Category")

# Sidebar dropdown for selecting mode
mode = st.sidebar.selectbox("Select Mode", ["Incident Details", "State Details"])

# Depending on mode, set sidebar title and create a summary container
if mode == "State Details":
    st.sidebar.title("State Details")
    summary_container = st.sidebar.container()
else:
    st.sidebar.title("Incident Details")
    summary_container = st.sidebar.container()  # Container created but not used in Incident Mode

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

#SQL query for filters
query = """
SELECT R.latitude, R.longitud, R.totinj, R.totkld, C.Accident_Type,
       R.narr1, R.narr2, R.narr3, R.narr4, R.narr5, 
       R.narr6, R.narr7, R.narr8, R.narr9, R.narr10, 
       R.narr11, R.narr12, R.narr13, R.narr14, R.narr15, R.state_name
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

#Adding filters
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

query += " LIMIT 100;"

try:
    filtered_data = pd.read_sql_query(query, conn)
except sqlite3.OperationalError as e:
    st.error(f"Error executing query: {e}")
    conn.close()
    st.stop()

conn.close()

#Combine narrative columns
def combine_narratives(row):
    narratives = [row[f"narr{i}"] for i in range(1, 16) if pd.notnull(row[f"narr{i}"])]
    return " ".join(narratives)

filtered_data["description"] = filtered_data.apply(combine_narratives, axis=1)

#Assign colors to accident types (original palette)
unique_accident_types = filtered_data["Accident_Type"].unique()
color_palette = ["blue", "green", "red", "orange", "purple", "cyan", "magenta"]
accident_colors = {
    accident_type: color_palette[i % len(color_palette)]
    for i, accident_type in enumerate(unique_accident_types)
}

#Sidebar for displaying selected marker details
incident_details_placeholder = st.sidebar.empty()

#Create the map
m = folium.Map(location=[37.0902, -95.7129], zoom_start=5, tiles=map_tiles)

if mode == "State Details":
    #State Mode: Add state markers
    state_markers = [
        {"lat": 34.0489, "lon": -111.0937, "state": "Arizona"},
        {"lat": 40.7128, "lon": -74.0060, "state": "New York"},
    ]
    for state in state_markers:
        folium.Marker(
            location=[state["lat"], state["lon"]],
            tooltip=f"State: {state['state']}"
        ).add_to(m)
else:
    #Incident Mode: Add incident markers and clustering
    if enable_clustering:
        marker_cluster = MarkerCluster()
        m.add_child(marker_cluster)

    marker_data = []
    for _, row in filtered_data.iterrows():
        accident_type = row["Accident_Type"]
        color = accident_colors.get(accident_type, "gray")
        description = row["description"]

        #Skip rows with missing latitude or longitude
        if pd.isnull(row["latitude"]) or pd.isnull(row["longitud"]):
            continue

        marker = folium.Marker(
            location=[row["latitude"], row["longitud"]],
            icon=folium.Icon(color=color),
            tooltip="Click for details"
        )

        if enable_clustering:
            marker_cluster.add_child(marker)
        else:
            m.add_child(marker)

        #Store marker data for sidebar display
        marker_data.append({
            "lat": row["latitude"],
            "lon": row["longitud"],
            "accident_type": accident_type,
            "description": description,
            "injuries": row["totinj"],
            "deaths": row["totkld"]
        })

    #Add heatmap if enabled
    if show_heatmap:
        heat_data = filtered_data[["latitude", "longitud"]].dropna().values.tolist()
        HeatMap(heat_data).add_to(m)

###############################################################################
# Make the map fill the available screen width and increase its height.
###############################################################################
clicked_data = st_folium(m, use_container_width=True, height=900)

with summary_container:
    if clicked_data and clicked_data.get("last_clicked"):
        lat = clicked_data["last_clicked"]["lat"]
        lon = clicked_data["last_clicked"]["lng"]

        if mode == "State Details":
            #In State Mode: Show details for clicked state
            clickedState = find_state(lat, lon)
            st.write(f"### Clicked State Coordinates: ({lat}, {lon})")
            st.write(f"### Clicked State : {clickedState}")

            wantedState = str(clickedState) 
            print(wantedState)
            state_data = filtered_data[filtered_data['state_name'] == wantedState]

            # 1. Total Number of Incidents
            total_incidents = state_data.shape[0]
            st.write(f"Total incidents in state {wantedState}: {total_incidents}")

            # 2. Average Train Speed During Incidents (if train_speed column exists)
            if 'train_speed' in state_data.columns:
                avg_speed = state_data['train_speed'].mean()
                st.write(f"Average train speed in state {wantedState}: {avg_speed:.2f} (units)")
            else:
                st.write("The train_speed column is not available in the data.")

            # 3. Number of Fatalities and Injuries
            total_fatalities = state_data['totkld'].sum()
            total_injuries = state_data['totinj'].sum()
            st.write(f"Total fatalities in state {wantedState}: {total_fatalities}")
            st.write(f"Total injuries in state {wantedState}: {total_injuries}")

            # 4. Most Common Weather Condition (if weather_condition column exists)
            if 'weather_condition' in state_data.columns:
                most_common_weather = (
                    state_data['weather_condition']
                    .value_counts()
                    .idxmax()
                )
                st.write(f"Most common weather condition in state {wantedState}: {most_common_weather}")
            else:
                st.write("The weather_condition column is not available in the data.")

            # 5. Incident Types Breakdown
            incident_breakdown = state_data['Accident_Type'].value_counts()
            st.write(f"Incident types breakdown in state {wantedState}:\n{incident_breakdown}")

            st.write("Placeholder for state-level details.")
        else:
            #In Incident Mode: Show details for clicked marker
            for marker in marker_data:
                if marker["lat"] == lat and marker["lon"] == lon:
                    with incident_details_placeholder.container():
                        st.write(f"### Accident Type: {marker['accident_type']}")
                        st.write(f"**Injuries:** {marker['injuries']}")
                        st.write(f"**Deaths:** {marker['deaths']}")
                        st.write(f"**Description:** {marker['description']}")

#Add Help Dropdown Menu
with st.expander("ℹ️ Help: How to Use the Railroad Incident Map"):
    st.markdown("""
    
    """)


#streamlit run "C:\Users\yongj\OneDrive\Desktop\Visualization Project\dynamic_filters_map.py" 
#Main issues: Currently, we dont have enough info, colours are shit, lack of looking good, and idk make it better lol
