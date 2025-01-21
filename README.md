# Manufacturing API 
 
## Setup Instructions 
 
1. Clone the repository: 
   cd manufacturing-api 
 
2. Create a virtual environment and install dependencies: 
   python -m venv env 
   env\Scripts\activate 
   pip install -r requirements.txt 
 
3. Run the application: 
   python app\main.py 
 
4. Test endpoints: 
   - Upload: POST /upload with a CSV file. 
   - Train: POST /train. 
   - Predict: POST /predict with JSON: 
       { "Temperature": 80, "Run_Time": 120 } 
 
5. Access the app at http://127.0.0.1:5000. 
