#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Traffic Accident Monitoring and Prediction System
Includes real-time monitoring, visualization, and historical analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import json
from typing import Dict, List, Tuple
import threading
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Traffic Accident Monitoring System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Keys
TOMTOM_KEY = "6G8wc5EIC3zgZ9zAGtvBOciDvO3Alrcl"
WEATHER_KEY = "e825edbdbe8eedb0ccb8b37c0f4d7e6f"
GOOGLE_PLACES_KEY = "AIzaSyCAdSZAmwIgv0bHB7OWKOa-GbcDDlYEs4E"  # TODO: Replace with your valid Google Places API key
MODEL_PATH = "C:\\Users\\Luo\\Desktop\\accident_severity_model.pkl"

# Data file path - San Francisco historical data
SF_ACCIDENT_DATA_FILE = "C:\\Users\\Luo\\Desktop\\sf_accidents.csv"

# San Francisco boundaries
SF_BOUNDS = {
    'min_lat': 37.6398,
    'max_lat': 37.8324,
    'min_lon': -122.5247,
    'max_lon': -122.3366
}

# Severity configuration
SEVERITY_CONFIG = {
    1: {'name': 'Minor', 'color': 'green', 'icon': '✓'},
    2: {'name': 'Moderate', 'color': 'yellow', 'icon': '⚠'},
    3: {'name': 'Severe', 'color': 'orange', 'icon': '⚠⚠'},
    4: {'name': 'Critical', 'color': 'red', 'icon': '🚨'}
}

class AccidentPredictor:
    """Accident severity predictor"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.label_encoders = {}
        self.load_model()
    
    def load_model(self):
        """Load prediction model"""
        try:
            model_data = joblib.load(MODEL_PATH)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.label_encoders = model_data.get('label_encoders', {})
            st.success("✅ Model loaded successfully")
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
    
    def get_weather_data(self, lat: float, lon: float) -> Dict:
        """Get weather data"""
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': WEATHER_KEY,
                'units': 'imperial'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'] * 0.02953,
                'visibility': data.get('visibility', 10000) * 0.000621371,
                'wind_speed': data['wind']['speed'],
                'wind_direction': self._degree_to_direction(data['wind'].get('deg', 0)),
                'weather_condition': data['weather'][0]['main'] if data.get('weather') else 'Unknown',
                'precipitation': data.get('rain', {}).get('1h', 0) * 0.0393701
            }
        except Exception as e:
            st.warning(f"Weather data retrieval failed: {e}")
            return self._get_default_weather()
    
    def _degree_to_direction(self, degree: float) -> str:
        """Convert degree to direction"""
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        index = int((degree + 22.5) / 45) % 8
        return directions[index]
    
    def _get_default_weather(self) -> Dict:
        """Default weather data"""
        return {
            'temperature': 60,
            'humidity': 50,
            'pressure': 30.0,
            'visibility': 10,
            'wind_speed': 5,
            'wind_direction': 'N',
            'weather_condition': 'Unknown',
            'precipitation': 0
        }
    
    def simplify_weather_condition(self, condition: str) -> str:
        """
        Simplify weather condition to match training data categories
        """
        if pd.isnull(condition) or condition == 'Unknown':
            return 'Unknown'
        
        condition = str(condition).lower()
        
        if any(word in condition for word in ['fair', 'clear']):
            return 'Clear'
        elif any(word in condition for word in ['cloud', 'overcast']):
            return 'Cloudy'
        elif 'rain' in condition and 'light' not in condition:
            return 'Rain'
        elif any(word in condition for word in ['light rain', 'drizzle']):
            return 'Light_Rain'
        elif 'snow' in condition:
            return 'Snow'
        elif any(word in condition for word in ['fog', 'mist']):
            return 'Fog'
        elif any(word in condition for word in ['thunder', 'storm']):
            return 'Storm'
        else:
            return 'Other'
    
    def get_road_features(self, lat: float, lon: float) -> Dict:
        """Get road features"""
        try:
            overpass_url = "http://overpass-api.de/api/interpreter"
            radius = 100
            
            query = f"""
            [out:json][timeout:25];
            (
              node["highway"="traffic_signals"](around:{radius},{lat},{lon});
              node["highway"="stop"](around:{radius},{lat},{lon});
              node["highway"="crossing"](around:{radius},{lat},{lon});
              node["highway"="give_way"](around:{radius},{lat},{lon});
              node["traffic_calming"](around:{radius},{lat},{lon});
              way["junction"="roundabout"](around:{radius},{lat},{lon});
              way["railway"](around:{radius},{lat},{lon});
              node["amenity"](around:{radius},{lat},{lon});
            );
            out body;
            """
            
            response = requests.get(overpass_url, 
                                  params={'data': query}, 
                                  timeout=30)
            response.raise_for_status()
            data = response.json()
            
            features = {
                'Amenity': False, 'Bump': False, 'Crossing': False,
                'Give_Way': False, 'Junction': False, 'No_Exit': False,
                'Railway': False, 'Roundabout': False, 'Station': False,
                'Stop': False, 'Traffic_Calming': False, 'Traffic_Signal': False,
                'Turning_Loop': False
            }
            
            for element in data['elements']:
                tags = element.get('tags', {})
                
                if 'highway' in tags:
                    highway_type = tags['highway']
                    if highway_type == 'traffic_signals':
                        features['Traffic_Signal'] = True
                    elif highway_type == 'stop':
                        features['Stop'] = True
                    elif highway_type == 'crossing':
                        features['Crossing'] = True
                    elif highway_type == 'give_way':
                        features['Give_Way'] = True
                
                if 'traffic_calming' in tags:
                    features['Traffic_Calming'] = True
                    if tags['traffic_calming'] == 'bump':
                        features['Bump'] = True
                
                if 'junction' in tags and tags['junction'] == 'roundabout':
                    features['Roundabout'] = True
                    features['Junction'] = True
                
                if 'railway' in tags:
                    features['Railway'] = True
                
                if 'amenity' in tags:
                    features['Amenity'] = True
            
            return features
            
        except Exception as e:
            st.warning(f"Road feature retrieval failed: {e}")
            return {key: False for key in [
                'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'
            ]}
    
    def prepare_features(self, weather: Dict, road_features: Dict) -> pd.DataFrame:
        """Prepare prediction features"""
        now = datetime.now()
        
        features = {}
        
        # Time features
        features['Hour'] = now.hour
        features['DayOfWeek'] = now.weekday()
        features['Month'] = now.month
        features['IsWeekend'] = 1 if now.weekday() >= 5 else 0
        features['IsRushHour'] = 1 if (7 <= now.hour <= 9) or (16 <= now.hour <= 18) else 0
        
        # Weather features
        features['Temperature(F)'] = weather['temperature']
        features['Humidity(%)'] = weather['humidity']
        features['Pressure(in)'] = weather['pressure']
        features['Visibility(mi)'] = min(weather['visibility'], 10)
        features['Wind_Speed(mph)'] = weather['wind_speed']
        features['Precipitation(in)'] = weather['precipitation']
        features['Precipitation_Missing'] = 0
        
        # Road features
        for feature_name, value in road_features.items():
            features[feature_name] = 1 if value else 0
        
        df = pd.DataFrame([features])
        
        # Handle encoded features
        if 'Wind_Direction' in self.label_encoders:
            wind_dir = weather['wind_direction']
            if wind_dir in self.label_encoders['Wind_Direction'].classes_:
                df['Wind_Direction_encoded'] = self.label_encoders['Wind_Direction'].transform([wind_dir])[0]
            else:
                df['Wind_Direction_encoded'] = 0
        
        if 'Weather_Condition' in self.label_encoders:
            weather_cond = self.simplify_weather_condition(weather['weather_condition'])
            
            if weather_cond in self.label_encoders['Weather_Condition'].classes_:
                df['Weather_Condition_encoded'] = self.label_encoders['Weather_Condition'].transform([weather_cond])[0]
            else:
                df['Weather_Condition_encoded'] = 0
        
        if 'Sunrise_Sunset' in self.label_encoders:
            is_night = now.hour < 6 or now.hour > 18
            sunrise_sunset = 'Night' if is_night else 'Day'
            if sunrise_sunset in self.label_encoders['Sunrise_Sunset'].classes_:
                df['Sunrise_Sunset_encoded'] = self.label_encoders['Sunrise_Sunset'].transform([sunrise_sunset])[0]
            else:
                df['Sunrise_Sunset_encoded'] = 0
        
        # Ensure correct feature order
        result_df = pd.DataFrame()
        for feature in self.feature_columns:
            if feature in df.columns:
                result_df[feature] = df[feature]
            else:
                result_df[feature] = 0
        
        return result_df
    
    def predict_severity(self, lat: float, lon: float) -> Tuple[int, Dict, Dict]:
        """Predict accident severity"""
        # Get weather and road data
        weather = self.get_weather_data(lat, lon)
        road_features = self.get_road_features(lat, lon)
        
        # Prepare features
        features_df = self.prepare_features(weather, road_features)
        
        # Predict
        severity = self.model.predict(features_df)[0]
        
        return severity, weather, road_features

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_accident_data():
    """Load historical accident data"""
    if os.path.exists(SF_ACCIDENT_DATA_FILE):
        try:
            # Read historical data
            df = pd.read_csv(SF_ACCIDENT_DATA_FILE)
            # Use flexible datetime parsing
            df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
            return df
        except Exception as e:
            st.error(f"Data loading failed: {e}")
            # Try other parsing methods
            try:
                df = pd.read_csv(SF_ACCIDENT_DATA_FILE)
                df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='ISO8601')
                return df
            except:
                try:
                    df = pd.read_csv(SF_ACCIDENT_DATA_FILE)
                    # Handle possible millisecond format
                    df['Start_Time'] = pd.to_datetime(df['Start_Time'].str[:19], format='%Y-%m-%d %H:%M:%S')
                    return df
                except Exception as e2:
                    st.error(f"Failed after trying multiple formats: {e2}")
                    return pd.DataFrame()
    else:
        st.error(f"Data file not found: {SF_ACCIDENT_DATA_FILE}")
        st.info("Please run filter_sf_accidents.py first to generate San Francisco data file")
        return pd.DataFrame()

def save_accident_data(accident_data):
    """Save new accident data to historical file"""
    try:
        new_row = {
            'ID': f"RT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{accident_data['id']}",
            'Start_Time': accident_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'End_Time': (accident_data['timestamp'] + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'Start_Lat': accident_data['lat'],
            'Start_Lng': accident_data['lon'],
            'End_Lat': accident_data['lat'],
            'End_Lng': accident_data['lon'],
            'Severity': accident_data['severity'],
            'Temperature(F)': accident_data['temperature'],
            'Humidity(%)': accident_data.get('humidity', 50),
            'Pressure(in)': accident_data.get('pressure', 30.0),
            'Visibility(mi)': accident_data['visibility'],
            'Wind_Speed(mph)': accident_data['wind_speed'],
            'Wind_Direction': accident_data.get('wind_direction', 'N'),
            'Weather_Condition': accident_data['weather_condition'],
            'Precipitation(in)': accident_data.get('precipitation', 0),
            'Amenity': accident_data.get('has_amenity', False),
            'Bump': accident_data.get('has_bump', False),
            'Crossing': accident_data.get('has_crossing', False),
            'Give_Way': accident_data.get('has_give_way', False),
            'Junction': accident_data.get('has_junction', False),
            'No_Exit': accident_data.get('has_no_exit', False),
            'Railway': accident_data.get('has_railway', False),
            'Roundabout': accident_data.get('has_roundabout', False),
            'Station': accident_data.get('has_station', False),
            'Stop': accident_data.get('has_stop', False),
            'Traffic_Calming': accident_data.get('has_traffic_calming', False),
            'Traffic_Signal': accident_data.get('has_signal', False),
            'Turning_Loop': accident_data.get('has_turning_loop', False),
            'Sunrise_Sunset': 'Night' if accident_data['timestamp'].hour < 6 or accident_data['timestamp'].hour > 18 else 'Day'
        }
        
        # Read existing data
        if os.path.exists(SF_ACCIDENT_DATA_FILE):
            df = pd.read_csv(SF_ACCIDENT_DATA_FILE)
        else:
            df = pd.DataFrame()
        
        # Append new data
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Save back to file
        df.to_csv(SF_ACCIDENT_DATA_FILE, index=False)
        
        # Clear cache to reload data
        st.cache_data.clear()
        
        return True
    except Exception as e:
        st.error(f"Data saving failed: {e}")
        return False

def get_tomtom_incidents():
    """Get TomTom traffic incidents"""
    try:
        base_url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
        bbox = f"{SF_BOUNDS['min_lon']},{SF_BOUNDS['min_lat']},{SF_BOUNDS['max_lon']},{SF_BOUNDS['max_lat']}"
        
        params = {
            'key': TOMTOM_KEY,
            'bbox': bbox,
            'fields': '{incidents{type,geometry{coordinates},properties{from,to,id,magnitudeOfDelay,length,delay}}}',
            'language': 'en-US',
            'categoryFilter': '0,1,2,3,4,5,6,7,8,9,10,11,14'
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        incidents = []
        
        if 'incidents' in data:
            for incident in data['incidents']:
                if incident['type'] in ['ACCIDENT', 'DANGEROUS_CONDITIONS']:
                    coords = incident['geometry']['coordinates'][0]
                    incidents.append({
                        'id': incident['properties']['id'],
                        'lat': coords[1],
                        'lon': coords[0],
                        'type': incident['type'],
                        'from': incident['properties'].get('from', ''),
                        'delay': incident['properties'].get('delay', 0)
                    })
        
        return incidents
        
    except Exception as e:
        st.error(f"TomTom data retrieval failed: {e}")
        return []

def create_accident_map(accidents_df, is_realtime=False):
    """Create accident map"""
    # Create map
    m = folium.Map(
        location=[37.7749, -122.4194],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Determine column names to use
    if is_realtime:
        lat_col, lon_col = 'lat', 'lon'
        time_col = 'timestamp'
        severity_col = 'severity'
        temp_col = 'temperature'
        weather_col = 'weather_condition'
        vis_col = 'visibility'
        desc_col = 'description'
    else:
        lat_col, lon_col = 'Start_Lat', 'Start_Lng'
        time_col = 'Start_Time'
        severity_col = 'Severity'
        temp_col = 'Temperature(F)'
        weather_col = 'Weather_Condition'
        vis_col = 'Visibility(mi)'
        desc_col = 'Description'
    
    # Add accident markers
    for _, row in accidents_df.iterrows():
        severity = int(row[severity_col])
        config = SEVERITY_CONFIG[severity]
        
        popup_html = f"""
        <b>Accident Information</b><br>
        Time: {row[time_col]}<br>
        Severity: {severity} ({config['name']})<br>
        Temperature: {row.get(temp_col, 'N/A')}°F<br>
        Weather: {row.get(weather_col, 'N/A')}<br>
        Visibility: {row.get(vis_col, 'N/A')} mi<br>
        {row.get(desc_col, '')}
        """
        
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=8 + severity * 2,
            popup=folium.Popup(popup_html, max_width=300),
            color=config['color'],
            fill=True,
            fillColor=config['color'],
            fillOpacity=0.6
        ).add_to(m)
    
    return m

def geocode_address(address):
    """Geocode address using Google Geocoding API"""
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': address,
            'key': GOOGLE_PLACES_KEY,
            'bounds': f"{SF_BOUNDS['min_lat']},{SF_BOUNDS['min_lon']}|{SF_BOUNDS['max_lat']},{SF_BOUNDS['max_lon']}"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data['status'] == 'OK' and data['results']:
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            return None, None
    except Exception as e:
        st.error(f"Geocoding failed: {e}")
        return None, None

def get_place_suggestions(query):
    """Get place suggestions from Google Places API"""
    try:
        url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        params = {
            'input': query,
            'key': GOOGLE_PLACES_KEY,
            'location': f"{37.7749},{-122.4194}",  # San Francisco center
            'radius': 20000,  # 20km radius
            'components': 'country:us',
            'types': 'geocode|establishment'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Debug information for API errors
        if data['status'] != 'OK':
            if data['status'] == 'REQUEST_DENIED':
                st.warning("Google Places API key is invalid or not properly configured. Please check your API key.")
                st.info("Make sure you have enabled Places API in Google Cloud Console and the API key has proper permissions.")
            return []
            
        if data['status'] == 'OK':
            suggestions = []
            for prediction in data['predictions']:
                # Only include suggestions that mention San Francisco or SF
                if any(term in prediction['description'].lower() for term in ['san francisco', 'sf, ca', 'california']):
                    suggestions.append(prediction['description'])
            return suggestions[:5]  # Return top 5 suggestions
        else:
            return []
    except Exception as e:
        st.error(f"Failed to get suggestions: {str(e)}")
        return []

def page_realtime_monitoring():
    """Page 1: Real-time monitoring"""
    st.title("🚨 Real-time Traffic Accident Monitoring")
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = AccidentPredictor()
    
    predictor = st.session_state.predictor
    
    # Auto-refresh settings
    col1, col2 = st.columns([3, 1])
    with col1:
        auto_refresh = st.checkbox("Auto-refresh (every 60 seconds)", key="auto_refresh")
    with col2:
        refresh_button = st.button("🔄 Refresh Now", key="manual_refresh")
    
    # Two main functional areas
    tab1, tab2 = st.tabs(["📡 TomTom Real-time Data", "📍 Manual Location Input"])
    
    with tab1:
        st.subheader("Get real-time traffic incidents from TomTom API")
        
        if refresh_button or auto_refresh:
            with st.spinner("Fetching traffic incident data..."):
                incidents = get_tomtom_incidents()
                
                if incidents:
                    st.success(f"Found {len(incidents)} traffic incidents")
                    
                    # Process each incident
                    progress_bar = st.progress(0)
                    saved_count = 0
                    
                    for idx, incident in enumerate(incidents):
                        progress_bar.progress((idx + 1) / len(incidents))
                        
                        # Predict severity
                        severity, weather, road_features = predictor.predict_severity(
                            incident['lat'], incident['lon']
                        )
                        
                        # Prepare data for saving
                        accident_data = {
                            'id': incident['id'],
                            'timestamp': datetime.now(),
                            'lat': incident['lat'],
                            'lon': incident['lon'],
                            'severity': severity,
                            'temperature': weather['temperature'],
                            'weather_condition': predictor.simplify_weather_condition(weather['weather_condition']),
                            'visibility': weather['visibility'],
                            'wind_speed': weather['wind_speed'],
                            'wind_direction': weather['wind_direction'],
                            'humidity': weather['humidity'],
                            'pressure': weather['pressure'],
                            'precipitation': weather['precipitation'],
                            'has_signal': road_features['Traffic_Signal'],
                            'has_junction': road_features['Junction'],
                            'has_crossing': road_features['Crossing'],
                            'has_amenity': road_features['Amenity'],
                            'has_bump': road_features['Bump'],
                            'has_give_way': road_features['Give_Way'],
                            'has_no_exit': road_features['No_Exit'],
                            'has_railway': road_features['Railway'],
                            'has_roundabout': road_features['Roundabout'],
                            'has_station': road_features['Station'],
                            'has_stop': road_features['Stop'],
                            'has_traffic_calming': road_features['Traffic_Calming'],
                            'has_turning_loop': road_features['Turning_Loop'],
                            'description': f"{incident['type']} - Delay: {incident['delay']}s"
                        }
                        
                        # Save to historical data
                        if save_accident_data(accident_data):
                            saved_count += 1
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Location", f"{incident['lat']:.4f}, {incident['lon']:.4f}")
                        with col2:
                            config = SEVERITY_CONFIG[severity]
                            st.metric("Predicted Severity", f"{config['icon']} {config['name']}")
                        with col3:
                            st.metric("Weather", weather['weather_condition'])
                    
                    progress_bar.empty()
                    st.info(f"Successfully saved {saved_count} new records to historical data")
                else:
                    st.info("No new traffic incidents detected")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(60)
            st.rerun()
    
    with tab2:
        st.subheader("Manually input accident location")
        
        # Initialize session state
        if 'manual_lat' not in st.session_state:
            st.session_state.manual_lat = 37.7749
        if 'manual_lon' not in st.session_state:
            st.session_state.manual_lon = -122.4194
        
        # Top section: Coordinate input and predict button
        st.markdown("### 📍 Location Coordinates")
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            # Use session state value for latitude
            lat = st.number_input(
                "Latitude", 
                min_value=37.6, 
                max_value=37.9, 
                value=st.session_state.manual_lat, 
                step=0.0001,
                key="lat_input",
                format="%.6f"
            )
            # Update session state when user manually changes the value
            if lat != st.session_state.manual_lat:
                st.session_state.manual_lat = lat
        with col2:
            # Use session state value for longitude
            lon = st.number_input(
                "Longitude", 
                min_value=-122.6, 
                max_value=-122.3, 
                value=st.session_state.manual_lon, 
                step=0.0001,
                key="lon_input",
                format="%.6f"
            )
            # Update session state when user manually changes the value
            if lon != st.session_state.manual_lon:
                st.session_state.manual_lon = lon
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            if st.button("🔮 Predict Severity", key="predict_manual_tab2", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    # Predict
                    severity, weather, road_features = predictor.predict_severity(lat, lon)
                    
                    # Prepare data for saving
                    accident_data = {
                        'id': f"MANUAL_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'timestamp': datetime.now(),
                        'lat': lat,
                        'lon': lon,
                        'severity': severity,
                        'temperature': weather['temperature'],
                        'weather_condition': predictor.simplify_weather_condition(weather['weather_condition']),
                        'visibility': weather['visibility'],
                        'wind_speed': weather['wind_speed'],
                        'wind_direction': weather['wind_direction'],
                        'humidity': weather['humidity'],
                        'pressure': weather['pressure'],
                        'precipitation': weather['precipitation'],
                        'has_signal': road_features['Traffic_Signal'],
                        'has_junction': road_features['Junction'],
                        'has_crossing': road_features['Crossing'],
                        'has_amenity': road_features['Amenity'],
                        'has_bump': road_features['Bump'],
                        'has_give_way': road_features['Give_Way'],
                        'has_no_exit': road_features['No_Exit'],
                        'has_railway': road_features['Railway'],
                        'has_roundabout': road_features['Roundabout'],
                        'has_station': road_features['Station'],
                        'has_stop': road_features['Stop'],
                        'has_traffic_calming': road_features['Traffic_Calming'],
                        'has_turning_loop': road_features['Turning_Loop'],
                        'description': 'Manually entered location'
                    }
                    
                    # Save to historical data
                    if save_accident_data(accident_data):
                        st.success("✅ Prediction completed and saved to historical data")
                    else:
                        st.warning("⚠️ Prediction completed but saving failed")
                    
                    # Store prediction results in session state to display below
                    st.session_state.show_prediction = True
                    st.session_state.prediction_results = {
                        'severity': severity,
                        'weather': weather,
                        'road_features': road_features
                    }
        
        st.markdown("---")
        
        # Address input section
        st.markdown("### 🔍 Or search by address")
        st.markdown("**Enter address or place name in San Francisco:**")
        
        # Initialize session state for address input
        if 'selected_address' not in st.session_state:
            st.session_state.selected_address = None
        if 'show_map' not in st.session_state:
            st.session_state.show_map = False
        
        # Address input with autocomplete
        address_query = st.text_input(
            "Start typing an address (e.g., 'Golden Gate', 'Market Street')", 
            placeholder="Type at least 3 characters to see suggestions",
            key="address_query_input"
        )
        
        # Get suggestions when user types
        if address_query and len(address_query) >= 3:
            with st.spinner("Getting suggestions..."):
                suggestions = get_place_suggestions(address_query)
                
            if suggestions:
                # Show suggestions in a selectbox
                selected_suggestion = st.selectbox(
                    "Select from suggestions:",
                    [""] + suggestions,
                    key="address_suggestions"
                )
                
                if selected_suggestion:
                    st.session_state.selected_address = selected_suggestion
                    address = selected_suggestion
                else:
                    address = address_query
            else:
                address = address_query
                st.info("No suggestions found. You can still proceed with your input.")
        else:
            address = None
            if address_query:
                st.info("Type at least 3 characters to see suggestions")
        
        # Process selected address
        if address and (st.button("🔍 Search Address") or st.session_state.selected_address):
            # Geocode the address
            with st.spinner("Locating address..."):
                map_lat, map_lon = geocode_address(address)
            
            if map_lat and map_lon:
                st.success(f"✅ Address found: {map_lat:.6f}, {map_lon:.6f}")
                st.session_state.show_map = True
                st.session_state.map_center = [map_lat, map_lon]
                st.session_state.map_address = address
            else:
                st.error("Could not find the address. Please try a different address.")
                st.session_state.show_map = False
        
        # Show map if address was found
        if st.session_state.get('show_map', False):
            st.markdown("---")
            st.subheader("📍 Click on the map to select location")
            st.info("💡 Click anywhere on the map to update the coordinates above")
            
            # Create map centered on geocoded location
            m = folium.Map(
                location=st.session_state.map_center, 
                zoom_start=16
            )
            
            # Add marker for geocoded location
            folium.Marker(
                st.session_state.map_center,
                popup=f"Searched: {st.session_state.map_address}",
                tooltip="Original location",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
            
            # Add click event handler
            m.add_child(folium.LatLngPopup())
            
            # Display map
            map_data = st_folium(
                m, 
                width=None, 
                height=400, 
                returned_objects=["last_object_clicked"],
                key=f"address_map_{datetime.now().timestamp()}"
            )
            
            # Check if map was clicked
            if map_data['last_object_clicked'] is not None:
                if 'lat' in map_data['last_object_clicked'] and 'lng' in map_data['last_object_clicked']:
                    clicked_lat = map_data['last_object_clicked']['lat']
                    clicked_lon = map_data['last_object_clicked']['lng']
                    
                    # Only update if coordinates are different
                    if (abs(clicked_lat - st.session_state.manual_lat) > 0.00001 or 
                        abs(clicked_lon - st.session_state.manual_lon) > 0.00001):
                        
                        # Update session state with clicked coordinates
                        st.session_state.manual_lat = clicked_lat
                        st.session_state.manual_lon = clicked_lon
                        
                        # Show success message
                        st.success(f"✅ Coordinates updated: {clicked_lat:.6f}, {clicked_lon:.6f}")
                        st.info("👆 Click the 'Predict Severity' button above to analyze this location")
                        
                        # Force rerun to update the number inputs
                        st.rerun()
        
        # Display prediction results if available
        if st.session_state.get('show_prediction', False):
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            results = st.session_state.prediction_results
            severity = results['severity']
            weather = results['weather']
            road_features = results['road_features']
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Prediction Results")
                config = SEVERITY_CONFIG[severity]
                st.metric("Severity", f"{config['icon']} {config['name']}")
                
                st.subheader("Weather Information")
                st.write(f"🌡️ Temperature: {weather['temperature']:.1f}°F")
                st.write(f"☁️ Weather: {weather['weather_condition']}")
                st.write(f"👁️ Visibility: {weather['visibility']:.1f} mi")
                st.write(f"💨 Wind Speed: {weather['wind_speed']:.1f} mph")
            
            with col2:
                st.subheader("Road Features")
                road_features_display = []
                if road_features['Traffic_Signal']:
                    road_features_display.append("🚦 Traffic Signal")
                if road_features['Junction']:
                    road_features_display.append("🔀 Junction")
                if road_features['Crossing']:
                    road_features_display.append("🚶 Pedestrian Crossing")
                if road_features['Stop']:
                    road_features_display.append("🛑 Stop Sign")
                if road_features['Railway']:
                    road_features_display.append("🚂 Railway")
                
                if road_features_display:
                    for feature in road_features_display:
                        st.write(feature)
                else:
                    st.write("No special road features")
            
            # Clear prediction flag after displaying
            st.session_state.show_prediction = False

def page_today_accidents():
    """Page 2: Today's accidents"""
    st.title("📊 Today's Traffic Accident Statistics")
    
    # Load data
    df = load_accident_data()
    
    if df.empty:
        st.warning("No data loaded")
        return
    
    # Filter today's data
    today = datetime.now().date()
    today_df = df[df['Start_Time'].dt.date == today].copy()
    
    if today_df.empty:
        st.warning("No traffic accidents recorded today")
        st.info("Tip: You can get new accident data through the real-time monitoring page")
        return
    
    # Upper part: Accident list
    st.subheader("📋 Today's Accident List")
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Accidents", len(today_df))
    with col2:
        avg_severity = today_df['Severity'].mean()
        st.metric("Average Severity", f"{avg_severity:.2f}")
    with col3:
        severe_count = len(today_df[today_df['Severity'] >= 3])
        st.metric("Severe Accidents", severe_count)
    with col4:
        latest_time = today_df['Start_Time'].max().strftime('%H:%M')
        st.metric("Last Update", latest_time)
    
    # Accident table
    display_columns = ['Start_Time', 'Severity', 'Start_Lat', 'Start_Lng', 
                      'Weather_Condition', 'Temperature(F)']
    
    # Check if Description column exists
    if 'Description' in today_df.columns:
        display_columns.append('Description')
    
    display_df = today_df[display_columns].copy()
    display_df['Start_Time'] = display_df['Start_Time'].dt.strftime('%H:%M:%S')
    display_df['Severity'] = display_df['Severity'].apply(
        lambda x: f"{SEVERITY_CONFIG[x]['icon']} {SEVERITY_CONFIG[x]['name']}"
    )
    
    # Rename columns
    new_column_names = ['Time', 'Severity', 'Latitude', 'Longitude', 'Weather', 'Temperature(°F)']
    if 'Description' in display_columns:
        new_column_names.append('Description')
    display_df.columns = new_column_names
    
    # Display only first 50 records
    st.dataframe(display_df.head(50), use_container_width=True, height=300)
    if len(display_df) > 50:
        st.info(f"Showing first 50 records out of {len(display_df)} total")
    
    # Lower part: Map visualization
    st.subheader("🗺️ Accident Location Map")
    
    # Create map
    accident_map = create_accident_map(today_df)
    
    # Display map
    map_data = st_folium(accident_map, width=None, height=500, returned_objects=["last_object_clicked"])
    
    # If marker clicked, show details
    if map_data['last_object_clicked']:
        st.info("Click on markers on the map to view details")
    
    # Statistics grouped by severity
    col1, col2 = st.columns(2)
    
    with col1:
        # Severity distribution pie chart
        severity_counts = today_df['Severity'].value_counts().sort_index()
        labels = [SEVERITY_CONFIG[i]['name'] for i in severity_counts.index]
        colors = [SEVERITY_CONFIG[i]['color'] for i in severity_counts.index]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=severity_counts.values,
            marker=dict(colors=colors),
            hole=0.3
        )])
        fig_pie.update_layout(title="Severity Distribution", height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Time distribution chart
        today_df['hour'] = today_df['Start_Time'].dt.hour
        hourly_counts = today_df.groupby('hour').size()
        
        fig_time = go.Figure(data=[go.Bar(
            x=hourly_counts.index,
            y=hourly_counts.values,
            marker_color='lightblue'
        )])
        fig_time.update_layout(
            title="Hourly Distribution",
            xaxis_title="Hour",
            yaxis_title="Number of Accidents",
            height=400
        )
        st.plotly_chart(fig_time, use_container_width=True)

def page_historical_analysis():
    """Page 3: Historical analysis"""
    st.title("📈 Historical Accident Analysis")
    
    # Load data
    df = load_accident_data()
    
    if df.empty:
        st.warning("No historical data loaded")
        return
    
    # Display data overview
    st.subheader("📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        date_range = f"{df['Start_Time'].min().strftime('%Y-%m-%d')} to {df['Start_Time'].max().strftime('%Y-%m-%d')}"
        st.metric("Date Range", date_range)
    with col3:
        # Count real-time data by checking ID prefix
        realtime_count = df[df['ID'].str.startswith('RT_', na=False)].shape[0]
        st.metric("Real-time Data", f"{realtime_count:,}")
    
    # Date range selection
    st.subheader("📅 Time Range Filter")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=df['Start_Time'].min().date())
    with col2:
        end_date = st.date_input("End Date", value=df['Start_Time'].max().date())
    
    # Filter data
    mask = (df['Start_Time'].dt.date >= start_date) & (df['Start_Time'].dt.date <= end_date)
    filtered_df = df[mask].copy()
    
    if filtered_df.empty:
        st.warning("No data in selected time range")
        return
    
    # Statistics
    st.subheader("📊 Statistical Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Accidents", f"{len(filtered_df):,}")
    with col2:
        st.metric("Average Severity", f"{filtered_df['Severity'].mean():.2f}")
    with col3:
        days = (end_date - start_date).days + 1
        st.metric("Daily Average", f"{len(filtered_df)/days:.1f}")
    with col4:
        severe_pct = len(filtered_df[filtered_df['Severity'] >= 3]) / len(filtered_df) * 100
        st.metric("Severe Accident %", f"{severe_pct:.1f}%")
    with col5:
        realtime_pct = filtered_df['ID'].str.startswith('RT_', na=False).sum() / len(filtered_df) * 100
        st.metric("Real-time Data %", f"{realtime_pct:.1f}%")
    
    # High-risk area identification method selection
    st.subheader("🎯 High-Risk Area Identification")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        method = st.selectbox(
            "Identification Method",
            ["Grid Aggregation", "DBSCAN Clustering"],
            help="Choose different algorithms to identify high-risk areas"
        )
    with col2:
        if method == "Grid Aggregation":
            grid_size = st.slider("Grid Precision", 2, 4, 3, 
                                 help="Decimal places, 3 means ~111m grid")
        elif method == "DBSCAN Clustering":
            eps = st.slider("Cluster Radius (degrees)", 0.001, 0.01, 0.003, 0.001,
                           help="~0.003 degrees = 333m")
            min_samples = st.slider("Minimum Samples", 3, 20, 5)
    with col3:
        weight_by_severity = st.checkbox("Weight by Severity", value=True)
    
    # Identify high-risk areas
    if method == "Grid Aggregation":
        # Grid statistics
        filtered_df['lat_round'] = filtered_df['Start_Lat'].round(grid_size)
        filtered_df['lon_round'] = filtered_df['Start_Lng'].round(grid_size)
        
        # Count each grid
        if weight_by_severity:
            # Weighted statistics (severity as weight)
            grid_stats = filtered_df.groupby(['lat_round', 'lon_round']).agg({
                'Severity': ['count', 'mean', 'sum']
            }).reset_index()
            grid_stats.columns = ['lat', 'lon', 'count', 'avg_severity', 'severity_sum']
            # Calculate danger index
            grid_stats['danger_index'] = grid_stats['count'] * grid_stats['avg_severity']
        else:
            grid_stats = filtered_df.groupby(['lat_round', 'lon_round']).agg({
                'Severity': ['count', 'mean']
            }).reset_index()
            grid_stats.columns = ['lat', 'lon', 'count', 'avg_severity']
            grid_stats['danger_index'] = grid_stats['count']
        
        # Find high-risk areas
        danger_threshold = grid_stats['danger_index'].quantile(0.9)
        danger_zones = grid_stats[
            (grid_stats['danger_index'] >= danger_threshold) |
            (grid_stats['avg_severity'] >= 3)
        ].sort_values('danger_index', ascending=False).head(20)
        
    elif method == "DBSCAN Clustering":
        from sklearn.cluster import DBSCAN
        
        # Prepare coordinate data
        coords = filtered_df[['Start_Lat', 'Start_Lng']].values
        
        # If weighting by severity, repeat coordinates of severe accidents
        if weight_by_severity:
            weighted_coords = []
            weights = []
            for idx, row in filtered_df.iterrows():
                weight = int(row['Severity'])
                for _ in range(weight):
                    weighted_coords.append([row['Start_Lat'], row['Start_Lng']])
                    weights.append(row['Severity'])
            coords = np.array(weighted_coords)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        # Analyze clustering results
        cluster_labels = clustering.labels_
        unique_clusters = set(cluster_labels) - {-1}  # Exclude noise points
        
        danger_zones_list = []
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_coords = coords[cluster_mask]
            
            # Calculate cluster center and statistics
            center_lat = cluster_coords[:, 0].mean()
            center_lon = cluster_coords[:, 1].mean()
            
            # Get accident info from original data
            cluster_accidents = filtered_df[
                (filtered_df['Start_Lat'].between(center_lat - eps, center_lat + eps)) &
                (filtered_df['Start_Lng'].between(center_lon - eps, center_lon + eps))
            ]
            
            danger_zones_list.append({
                'lat': center_lat,
                'lon': center_lon,
                'count': len(cluster_accidents),
                'avg_severity': cluster_accidents['Severity'].mean(),
                'danger_index': len(cluster_accidents) * cluster_accidents['Severity'].mean(),
                'radius': eps * 111000  # Convert to meters
            })
        
        danger_zones = pd.DataFrame(danger_zones_list).sort_values('danger_index', ascending=False).head(20)
    
    # Create map
    st.subheader("🗺️ High-Risk Areas Map")
    
    # Create base map
    m = folium.Map(
        location=[37.7749, -122.4194],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add heatmap layer
    from folium.plugins import HeatMap
    
    # Prepare heatmap data
    if weight_by_severity:
        heat_data = [[row['Start_Lat'], row['Start_Lng'], row['Severity']] 
                     for _, row in filtered_df.iterrows()]
    else:
        heat_data = [[row['Start_Lat'], row['Start_Lng'], 1] 
                     for _, row in filtered_df.iterrows()]
    
    HeatMap(heat_data, min_opacity=0.2, radius=15, blur=10).add_to(m)
    
    # Add high-risk area markers
    if not danger_zones.empty:
        for idx, zone in danger_zones.iterrows():
            # Determine color (based on danger index)
            if zone['danger_index'] > danger_zones['danger_index'].quantile(0.8):
                color = 'darkred'
                fill_opacity = 0.5
            elif zone['danger_index'] > danger_zones['danger_index'].quantile(0.6):
                color = 'red'
                fill_opacity = 0.4
            else:
                color = 'orange'
                fill_opacity = 0.3
            
            # Calculate radius
            if method == "DBSCAN Clustering":
                radius = zone.get('radius', 200)
            else:
                radius = 300  # Grid method uses fixed radius
            
            # Add circle marker
            folium.Circle(
                location=[zone['lat'], zone['lon']],
                radius=radius,
                popup=folium.Popup(
                    f"""<b>High-Risk Area</b><br>
                    Location: ({zone['lat']:.4f}, {zone['lon']:.4f})<br>
                    Accidents: {int(zone['count'])}<br>
                    Avg Severity: {zone['avg_severity']:.2f}<br>
                    Danger Index: {zone['danger_index']:.1f}""",
                    max_width=300
                ),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=fill_opacity
            ).add_to(m)
            
            # Add labels (show only top 10)
            if idx < 10:
                folium.Marker(
                    location=[zone['lat'], zone['lon']],
                    icon=folium.DivIcon(
                        html=f"""<div style="font-size: 12px; color: white; 
                                background-color: {color}; padding: 2px 5px; 
                                border-radius: 3px; font-weight: bold;">
                                #{idx+1}</div>"""
                    )
                ).add_to(m)
    
    # Display map
    st_folium(m, width=None, height=600)
    
    # High-risk area detailed list
    if not danger_zones.empty:
        st.subheader("📋 High-Risk Area Rankings")
        
        # Prepare display data
        danger_zones_display = danger_zones.copy()
        danger_zones_display['Rank'] = range(1, len(danger_zones_display) + 1)
        danger_zones_display['Location'] = danger_zones_display.apply(
            lambda x: f"({x['lat']:.4f}, {x['lon']:.4f})", axis=1
        )
        danger_zones_display['Severity'] = danger_zones_display['avg_severity'].apply(
            lambda x: f"{x:.2f} {SEVERITY_CONFIG[int(round(x))]['icon']}"
        )
        danger_zones_display['Risk Level'] = danger_zones_display['danger_index'].apply(
            lambda x: '🔴 Very High' if x > danger_zones['danger_index'].quantile(0.8) 
            else '🟠 High' if x > danger_zones['danger_index'].quantile(0.6) 
            else '🟡 Medium'
        )
        
        display_columns = ['Rank', 'Location', 'count', 'Severity', 'Risk Level']
        danger_zones_display = danger_zones_display[display_columns]
        danger_zones_display.columns = ['Rank', 'Location', 'Accident Count', 'Avg Severity', 'Risk Level']
        
        st.dataframe(
            danger_zones_display.set_index('Rank'),
            use_container_width=True,
            height=400
        )
    
    # Analysis charts
    st.subheader("📊 In-depth Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Severity distribution
        severity_dist = filtered_df['Severity'].value_counts().sort_index()
        
        fig_severity = go.Figure(data=[go.Bar(
            x=[SEVERITY_CONFIG[i]['name'] for i in severity_dist.index],
            y=severity_dist.values,
            marker_color=[SEVERITY_CONFIG[i]['color'] for i in severity_dist.index]
        )])
        
        fig_severity.update_layout(
            title='Severity Distribution',
            xaxis_title='Severity Level',
            yaxis_title='Number of Accidents',
            height=400
        )
        st.plotly_chart(fig_severity, use_container_width=True)
    
    with col2:
        # Time trend (by month)
        filtered_df['year_month'] = filtered_df['Start_Time'].dt.to_period('M')
        monthly_stats = filtered_df.groupby('year_month')['Severity'].agg(['count', 'mean'])
        monthly_stats.index = monthly_stats.index.to_timestamp()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_stats.index,
            y=monthly_stats['count'],
            mode='lines+markers',
            name='Accident Count',
            line=dict(color='blue', width=2)
        ))
        
        fig_trend.update_layout(
            title='Monthly Accident Trend',
            xaxis_title='Month',
            yaxis_title='Number of Accidents',
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Download high-risk area data
    if not danger_zones.empty and st.button("💾 Export High-Risk Area Data"):
        # Prepare export data
        export_data = danger_zones.copy()
        export_data['identified_time'] = datetime.now()
        export_data['data_range'] = f"{start_date} to {end_date}"
        export_data['total_accidents_analyzed'] = len(filtered_df)
        
        # Save to CSV
        export_filename = f"danger_zones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_data.to_csv(export_filename, index=False)
        st.success(f"✅ High-risk area data exported to: {export_filename}")

# Main program
def main():
    # Sidebar
    with st.sidebar:
        st.title("🚨 Traffic Accident Monitoring System")
        st.markdown("---")
        
        # Page selection
        page = st.radio(
            "Select Page",
            ["Real-time Monitoring", "Today's Statistics", "Historical Analysis"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.caption("System Description")
        st.caption("• Real-time traffic accident retrieval")
        st.caption("• Accident severity prediction")
        st.caption("• High-risk area analysis")
        
        st.markdown("---")
        
        # Display data file info
        if os.path.exists(SF_ACCIDENT_DATA_FILE):
            file_size = os.path.getsize(SF_ACCIDENT_DATA_FILE) / 1024 / 1024
            st.caption(f"Data file: {file_size:.2f} MB")
        
        st.caption(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display page based on selection
    if page == "Real-time Monitoring":
        page_realtime_monitoring()
    elif page == "Today's Statistics":
        page_today_accidents()
    elif page == "Historical Analysis":
        page_historical_analysis()

if __name__ == "__main__":
    main()