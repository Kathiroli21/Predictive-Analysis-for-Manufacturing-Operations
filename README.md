# Manufacturing API

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone repository-url
   cd manufacturing-api
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv env
   env\Scripts\activate 
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```
   python app/main.py
   ```

4. **Test endpoints:**
   - **Upload:** POST `/upload` with a CSV file.
   - **Train:** POST `/train`
 
    ```
     You can get the test data file `Machine Downtime.csv` from the 'test_data' folder,  
     which can be used for testing the endpoints.
     ```
   - **Predict:** POST `/predict` with JSON payload:
     ```json
     {
       "Hydraulic_Pressure(bar)": 30.5,
       "Coolant_Pressure(bar)": 5.8,
       "Air_System_Pressure(bar)": 2.1,
       "Coolant_Temperature": 45.0,
       "Hydraulic_Oil_Temperature(?C)": 50.2,
       "Spindle_Bearing_Temperature(?C)": 38.9,
       "Spindle_Vibration(?m)": 0.02,
       "Tool_Vibration(?m)": 0.03,
       "Spindle_Speed(RPM)": 1200,
       "Voltage(volts)": 220,
       "Torque(Nm)": 40,
       "Cutting(kN)": 15
     }
     ```



## API Endpoints

- **POST /upload**: Uploads a CSV file containing manufacturing data.
- **POST /train**: Trains the machine learning model using the latest uploaded data.
- **POST /predict**: Predicts downtime based on input parameters.
  
