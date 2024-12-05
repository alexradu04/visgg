import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import sqlite3

#Set Streamlit page config
st.set_page_config(layout="wide", page_title="Dynamic Railroad Incident Map")

#Initialize theme toggle
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

def toggle_theme():
    st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"

#Add a button to toggle themes
if st.sidebar.button("Toggle Dark/Light Theme"):
    toggle_theme()

#Set map tiles based on the selected theme
map_tiles = "CartoDB dark_matter" if st.session_state["theme"] == "dark" else "CartoDB positron"

#Path to SQLite database
sqlite_db = "C:\\Users\\yongj\\OneDrive\\Desktop\\Visualization Project\\Railroad Incidents\\railroad_incidents_cleaned_with_underscores.db"

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

#Sidebar dropdown for selecting mode
mode = st.sidebar.selectbox("Select Mode", ["Incident Details", "State Details"])

#ilters on the sidebar
st.sidebar.title("Dynamic Filters")
selected_speed = st.sidebar.selectbox("Speed Category", ["All"] + speed_categories)
selected_weather = st.sidebar.selectbox("Weather Condition", ["All"] + weather_conditions)
selected_year = st.sidebar.selectbox("Year Group", ["All"] + year_groups)
selected_death = st.sidebar.selectbox("Death Category", ["All"] + death_categories)
selected_injury = st.sidebar.selectbox("Injury Category", ["All"] + injury_categories)
selected_damage = st.sidebar.selectbox("Damage Category", ["All"] + damage_categories)

enable_clustering = st.sidebar.checkbox("Enable Clustering", value=True)
show_heatmap = st.sidebar.checkbox("Show Heatmap")

#SQL query for filters
query = """
SELECT R.latitude, R.longitud, R.totinj, R.totkld, C.Accident_Type,
       R.narr1, R.narr2, R.narr3, R.narr4, R.narr5, 
       R.narr6, R.narr7, R.narr8, R.narr9, R.narr10, 
       R.narr11, R.narr12, R.narr13, R.narr14, R.narr15
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

#Assign colors to accident types
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
    st.sidebar.title("State Details")
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
    st.sidebar.title("Incident Details")
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

    #Add heatmap if enabled//Need to be improved
    if show_heatmap:
        heat_data = filtered_data[["latitude", "longitud"]].dropna().values.tolist()
        HeatMap(heat_data).add_to(m)

#Render the map in Streamlit
clicked_data = st_folium(m, width=1100, height=800)

#Handle click events in both modes
if clicked_data and clicked_data.get("last_object_clicked"):
    lat = clicked_data["last_object_clicked"]["lat"]
    lon = clicked_data["last_object_clicked"]["lng"]

    if mode == "State Details":
        #In State Mode: Show placeholder details for clicked state
        st.sidebar.write(f"### Clicked State Coordinates: ({lat}, {lon})")
        st.sidebar.write("Placeholder for state-level details.")
    else:
        #In Incident Mode: Show details for clicked marker in the sidebar
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

#streamlit run "C:\Users\yongj\OneDrive\Desktop\Visualization Project\dynamic_filters_map.py" change asccortidng to directory
#Main issues: Currently, we dont have enough info, colours are shit, lack of looking good, and idk make it better lol