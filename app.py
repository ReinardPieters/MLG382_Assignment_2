import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import base64
import io
import torch
import os
from ft_transformer_model import FTTransformer
                              
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / 'data'

# Load models and encoders using the dynamic paths
ft_model = torch.load(data_dir / 'ft_transformer_model.pkl', map_location=torch.device('cpu'))

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
                {'label': 'clear', 'value': 'clear'},
                {'label': 'cloudy', 'value': 'cloudy'},
                {'label': 'overcast', 'value': 'overcast'},
                {'label': 'partly cloudy', 'value': 'partly cloudy'}
            ],
            placeholder='Select Cloud Cover',
            style={'width': '100%'}
            ),
        dcc.Dropdown(
            id='season',
            options=[
                {'label': 'Autumn', 'value': 'Autumn'},
                {'label': 'Spring', 'value': 'Spring'},
                {'label': 'Summer', 'value': 'Summer'},
                {'label': 'Winter', 'value': 'Winter'}
            ],
            placeholder='Select Season',
            style={'width': '100%'}
        ),
        dcc.Dropdown(
            id='location',
            options=[
                {'label': 'coastal', 'value': 'coastal'},
                {'label': 'inland', 'value': 'inland'},
                {'label': 'mountain', 'value': 'mountain'}
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
    html.Div(id='prediction-output', style={'marginTop': '30px', 'textAlign': 'center', 'fontSize': '18px'}),
    
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

        ft_input_cat = processed_input[['Season', 'Location']]
        ft_input_num = processed_input[['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Cloud Cover', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)', 'Temperature_Humidity', 'Wind_Speed_Precip']]
        
        #print("Numerical columns at inference:", ft_input_num.shape)

        # Make predictions using each model
        rf_prediction = rf_model.predict(processed_input)[0]
        svm_prediction = svm_model.predict(processed_input)[0]
        xgb_prediction = xgb_model.predict(processed_input)[0]
        ft_input_cat = torch.tensor(ft_input_cat.values, dtype=torch.long)
        ft_input_num = torch.tensor(ft_input_num.values, dtype=torch.float32)

                # Define model hyperparameters exactly as used during training
        cat_dims = [4, 3]  # Season, Location
        num_cont = 10      # Number of numeric columns
        emb_dim = 64
        transformer_layers = 4
        n_classes = 4      # Number of classes for "Weather Type"

        # Create the model instance
        ft_model = FTTransformer(
            cat_dims=cat_dims,
            num_cont=num_cont,
            emb_dim=emb_dim,
            transformer_layers=transformer_layers,
            n_classes=n_classes
        )

        # Load the trained weights
        model_path = os.path.join("data", "ft_transformer_model.pkl")
        ft_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

        # Set model to evaluation mode
        ft_model.eval()
        
        # Get prediction from the deep learning model
        with torch.no_grad():
            dl_output = ft_model(ft_input_cat, ft_input_num)
            if dl_output.dim() == 1:  # in case batch size is 1 and model returns shape [num_classes]
                dl_output = dl_output.unsqueeze(0)
            dl_prediction = torch.argmax(dl_output, dim=1).item()

        # Combine predictions
        predictions = {
            'Random Forest': weather_type_encoder.inverse_transform([rf_prediction])[0],
            'SVM': weather_type_encoder.inverse_transform([svm_prediction])[0],
            'XGBoost': weather_type_encoder.inverse_transform([xgb_prediction])[0],
            'FTtransformer': weather_type_encoder.inverse_transform([dl_prediction])[0]
        }

        # Display the predictions
        return (
            f"Predicted Weather Type:\n"
            f"- Random Forest: {predictions['Random Forest']}\n"
            f"- SVM: {predictions['SVM']}\n"
            f"- XGBoost: {predictions['XGBoost']}\n"
            f"- FTtransformer: {predictions['FTtransformer']}\n"
        )
    
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
server = app.server
# Run the app
if __name__ == '__main__':
    app.run(debug=True)