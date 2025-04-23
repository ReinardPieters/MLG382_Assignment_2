import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / 'data'

# Load models and encoders using the dynamic paths
with open(data_dir / 'rf_model.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open(data_dir / 'svm_model.pkl', 'rb') as svm_file:
    svm_model = pickle.load(svm_file)

with open(data_dir / 'xgb_model.pkl', 'rb') as xgb_file:
    xgb_model = pickle.load(xgb_file)

with open(data_dir / 'label_encoders.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)

with open(data_dir / 'scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

cover_encoder = encoders['cover_encoder']

season_encoder = encoders['season_encoder']

location_encoder = encoders['location_encoder']

weather_type_encoder = encoders['weather_type_encoder']

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Weather Type Prediction", style={'textAlign': 'center', 'marginBottom': '30px'}),

    html.Div([
        # Numeric inputs (first row)
        html.Div([
            dcc.Input(id='temperature', type='number', placeholder='Temperature (°C)', debounce=True, style={'width': '100%'}),
            dcc.Input(id='humidity', type='number', placeholder='Humidity (%)', debounce=True, style={'width': '100%'}),
            dcc.Input(id='wind_speed', type='number', placeholder='Wind Speed (km/h)', debounce=True, style={'width': '100%'})
        ], style={'display': 'grid', 'gap': '10px', 'gridTemplateColumns': 'repeat(3, 1fr)'}),

        html.Br(),

        # More numeric inputs (second row)
        html.Div([
            dcc.Input(id='precipitation', type='number', placeholder='Precipitation (%)', debounce=True, style={'width': '100%'}),
            dcc.Input(id='pressure', type='number', placeholder='Pressure (hPa)', debounce=True, style={'width': '100%'}),
            dcc.Input(id='uv_index', type='number', placeholder='UV Index', debounce=True, style={'width': '100%'})
        ], style={'display': 'grid', 'gap': '10px', 'gridTemplateColumns': 'repeat(3, 1fr)'}),

        html.Br(),

        # Dropdowns grouped together (third row)
        html.Div([
            dcc.Dropdown(
                id='cloud_cover',
                options=[
                    {'label': 'clear', 'value': 0},
                    {'label': 'cloudy', 'value': 1},
                    {'label': 'overcast', 'value': 2},
                    {'label': 'partly cloudy', 'value': 3}
                ],
                placeholder='Select Cloud Cover',
                style={'width': '100%'}
            ),
            dcc.Dropdown(
                id='season',
                options=[
                    {'label': 'Autumn', 'value': 0},
                    {'label': 'Spring', 'value': 1},
                    {'label': 'Summer', 'value': 2},
                    {'label': 'Winter', 'value': 3}
                ],
                placeholder='Select Season',
                style={'width': '100%'}
            ),
            dcc.Dropdown(
                id='location',
                options=[
                    {'label': 'coastal', 'value': 0},
                    {'label': 'inland', 'value': 1},
                    {'label': 'mountain', 'value': 2}
                ],
                placeholder='Select Location',
                style={'width': '100%'}
            )
        ], style={'display': 'grid', 'gap': '10px', 'gridTemplateColumns': 'repeat(3, 1fr)'}),

        html.Br(),

        # Visibility input on its own row
        html.Div([
            dcc.Input(id='visibility', type='number', placeholder='Visibility (km)', debounce=True, style={'width': '100%'})
        ], style={'marginTop': '10px'})
    ], style={
        'maxWidth': '900px',
        'margin': '0 auto',
        'padding': '20px',
        'border': '1px solid #ccc',
        'borderRadius': '10px',
        'boxShadow': '0 4px 10px rgba(0,0,0,0.05)',
        'backgroundColor': '#f9f9f9'
    }),

    html.Br(),

    # Predict button
    html.Div([
        html.Button('Predict Weather Type', id='predict-button', n_clicks=0, style={
            'padding': '10px 20px',
            'fontSize': '16px',
            'cursor': 'pointer',
            'borderRadius': '8px',
            'backgroundColor': '#007bff',
            'color': '#fff',
            'border': 'none'
        })
    ], style={'textAlign': 'center'}),

    # Output section
    html.Div(id='prediction-output', style={'marginTop': '30px', 'textAlign': 'center', 'fontSize': '18px'})
])
# Function to get the prediction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('temperature', 'value'),
     Input('humidity', 'value'),
     Input('wind_speed', 'value'),
     Input('precipitation', 'value'),
     Input('cloud_cover', 'value'),
     Input('pressure', 'value'),
     Input('uv_index', 'value'),
     Input('season', 'value'),
     Input('visibility', 'value'),
     Input('location', 'value')]
)
def predict_weather_type(n_clicks, temperature, humidity, wind_speed, precipitation, cloud_cover, pressure, uv_index, season, visibility, location):
    if n_clicks > 0:
        # Create a dictionary to hold the inputs
        inputs = {
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'precipitation': precipitation,
            'cloud_cover': cloud_cover,
            'pressure': pressure,
            'uv_index': uv_index,
            'season': season,
            'visibility': visibility,
            'location': location
        }

        # Preprocess the input data (now passing the dictionary)
        processed_input = preprocess_input(inputs)

        # Make predictions using each model
        rf_prediction = rf_model.predict(processed_input)[0]
        svm_prediction = svm_model.predict(processed_input)[0]
        xgb_prediction = xgb_model.predict(processed_input)[0]

        # Combine predictions
        predictions = {
            'Random Forest': weather_type_encoder.inverse_transform([rf_prediction])[0],
            'SVM': weather_type_encoder.inverse_transform([svm_prediction])[0],
            'XGBoost': weather_type_encoder.inverse_transform([xgb_prediction])[0]
        }

        # Display the predictions
        return f"Predicted Weather Type: {predictions['Random Forest']} (RF), {predictions['SVM']} (SVM), {predictions['XGBoost']} (XGBoost)"
    
    return ''

def preprocess_input(inputs):
    # Get the input values
    temp = inputs['temperature']
    humidity = inputs['humidity']
    wind_speed = inputs['wind_speed']
    precipitation = inputs['precipitation']
    cloud_cover = inputs['cloud_cover']
    pressure = inputs['pressure']
    uv_index = inputs['uv_index']
    season = inputs['season']
    visibility = inputs['visibility']
    location = inputs['location']
    
    # Create the additional features: Temperature_Humidity, Wind_Speed_Precip
    temperature_humidity = temp * humidity
    wind_speed_precip = wind_speed * precipitation
    
    # Handle encoding for non-numeric features
    cloud_cover = cover_encoder.transform([cloud_cover])[0]
    season = season_encoder.transform([season])[0]
    location = location_encoder.transform([location])[0]
    
    # Combine numeric and encoded features into a DataFrame
    input_data = pd.DataFrame([[temp, humidity, wind_speed, precipitation, cloud_cover, pressure, uv_index, season, visibility, location, temperature_humidity, wind_speed_precip]],
                              columns=['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Cloud Cover', 'Atmospheric Pressure', 'UV Index', 'Season', 'Visibility (km)', 'Location', 'Temperature_Humidity', 'Wind_Speed_Precip'])
    
    # Separate numeric columns for scaling
    numerical_columns = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 
                         'Atmospheric Pressure', 'UV Index', 'Visibility (km)', 
                         'Temperature_Humidity', 'Wind_Speed_Precip']
    
    # Scale the numeric features only
    input_data_numeric = input_data[numerical_columns]
    input_data_scaled = scaler.transform(input_data_numeric)
    
    # Combine scaled numeric features with the encoded categorical features
    input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=numerical_columns)
    input_data_final = pd.concat([input_data_scaled_df, 
                                  input_data[['Cloud Cover', 'Season', 'Location']].reset_index(drop=True)], axis=1)
    
    
    input_data_final = input_data_final[['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Cloud Cover', 'Atmospheric Pressure', 'UV Index', 'Season', 'Visibility (km)', 'Location', 'Temperature_Humidity', 'Wind_Speed_Precip']]
    
    return input_data_final

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
