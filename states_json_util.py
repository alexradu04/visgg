import json
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
file = open("C:\\dev\\visgg\\data\\us-states.json")
data = json.load(file)
# print(data['features'])
stateDict = {}
for feature in data['features']:
    stateID = feature['id']
    poligonType = feature['geometry']['type']
    if poligonType == 'Polygon':
        stateDict[stateID] = {"name": feature['properties']['name'], "poligons": [feature['geometry']['coordinates'][0]]}
        # print(feature['geometry']['coordinates'][0])
    elif poligonType == 'MultiPolygon':
        stateDict[stateID] = {"name": feature['properties']['name'], "poligons": feature['geometry']['coordinates'][0]}
        
# for key in stateDict:
#     print(key, stateDict[key])
#     print("\n\n")

def is_point_in_polygon(lat, lon, poligon):
    """Check if a point is inside a polygon"""
    n = len(poligon)
    inside = False
    p1x, p1y = poligon[0]
    for i in range(1, n + 1):
        p2x, p2y = poligon[i % n]
        if lat > min(p1y, p2y):
            if lat <= max(p1y, p2y):
                if lon <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or lon <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def find_state(lat, lon):
    """Find the state name for a given latitude and longitude"""
    for state_id, state_info in stateDict.items():
        for poligon in state_info['poligons']:
            if is_point_in_polygon(lat, lon, poligon):
                return state_info['name']
    return "Unknown"
    
#TESTS
"""


print(find_state(33.4484, -112.0740)) #Phoenix, Arizona
print(find_state(40.7128, -74.0060)) #New York City, New York
print(find_state(37.7749, -122.4194)) #San Francisco, California

"""