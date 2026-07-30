import os
import sys
import joblib
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore') # ignore warnings to keep terminal clean

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'outputs', 'best_parkinsons_model.pkl')
data_path = os.path.join(BASE_DIR, 'data', 'parkinsons.csv')

def main():
    # checking if the model is trained yet
    if not os.path.exists(model_path):
        print("Model not found! Please run train_parkinsons.py first to train the model.")
        sys.exit(1)
        
    print("\n--- Parkinson's Disease Prediction System ---")
    print("Because Parkinson's diagnosis requires 22 complex voice metrics (like MDVP:Jitter),")
    print("we cannot type them by hand.")
    
    try:
        df = pd.read_csv(data_path)
    except Exception:
        print(f"Error: Could not find dataset at {data_path}")
        sys.exit(1)
        
    patient_input = input("\nEnter a specific Patient ID (e.g. phon_R01_S01_1) or press ENTER for a random patient: ").strip()
    
    if patient_input == "":
        # Pick a random patient
        random_index = random.randint(0, len(df)-1)
        patient = df.iloc[random_index]
    else:
        # Try to find the specific patient
        matching_patients = df[df['name'] == patient_input]
        if matching_patients.empty:
            print(f"Error: Could not find patient '{patient_input}' in the database.")
            sys.exit(1)
        patient = matching_patients.iloc[0]
    patient_id = patient['name']
    actual_status = patient['status']
    
    # Extract only the 22 medical features
    patient_features = patient.drop(['name', 'status']).to_frame().T
    
    print("\nFetching vocal analysis records...")
    print(f"Patient ID: {patient_id}")
    print(f"Sample Metrics: Jitter={patient['MDVP:Jitter(%)']:.5f}, Shimmer={patient['MDVP:Shimmer']:.5f}, NHR={patient['NHR']:.5f}")
    
    print("\nAnalyzing vocal biomarkers...")
    
    # load the best model I saved earlier
    pipeline = joblib.load(model_path)
    
    try:
        # predict the status
        prediction = pipeline.predict(patient_features)[0]
        
        print("\n--- Diagnostic Results ---")
        if prediction == 1:
            print("[!] AI Diagnosis: POSITIVE for Parkinson's Disease")
        else:
            print("[+] AI Diagnosis: NEGATIVE (Healthy)")
            
        print("--------------------------")
        # For our testing purposes, print the real answer
        print(f"(Actual Ground Truth in database was: {'Positive' if actual_status == 1 else 'Negative'})")
        
    except Exception as e:
        print(f"\nError predicting status: {e}")

if __name__ == "__main__":
    main()
