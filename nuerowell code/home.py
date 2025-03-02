import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import io
import pyminizip

# Google Sheets API credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("neurowell-5b8eaaee5d15.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1lRSSzi6IfIEEKkfpMj67QI3S4ukyKzr2_yDlVEFGQcc/edit?gid=0#gid=0").sheet1

# Function to get all data from Google Sheets
def get_all_data():
    records = sheet.get_all_records()
    df = pd.DataFrame(records)
    return df

# Function to add a new patient
def add_new_patient(name, age, gender):
    sheet.append_row([name, age, gender, "", "", "", ""])

# Function to get patient data by name
def get_patient_data(name):
    records = get_all_data()
    return records[records['name'] == name]

def app():
    gradient_text_html = """
          <style>
          .gradient-text {
              font-weight: bold;
              background: -webkit-linear-gradient(left, #07539e, #4fc3f7, #ffffff);
              background: linear-gradient(to right, #07539e, #4fc3f7, #ffffff);
              -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;
              display: inline;
              font-size: 2.9em;
          }
          </style>
          <div class="gradient-text">Patient Management System</div>
          """

    st.markdown(gradient_text_html, unsafe_allow_html=True)
    st.image('divider.png')
    
    col1, col2, col3 = st.columns([1, 0.15, 1])
    
    with col1:
        option = st.selectbox("Select an option", ["New Patient", "Existing Patient"])

        if option == "New Patient":
            st.subheader("Add New Patient")
            name = st.text_input("Name")
            age = st.text_input("Age")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            if st.button("Add Patient"):
                add_new_patient(name, age, gender)
                st.success(f"Patient {name} added successfully!")

        elif option == "Existing Patient":
            st.subheader("Retrieve Patient Information")
            name = st.text_input("Enter Patient Name")
            if st.button("Get Patient Info"):
                patient_data = get_patient_data(name)
                if not patient_data.empty:
                    st.write(patient_data)
                else:
                    st.error("Patient not found!")

    with col3:
        st.subheader("All Patients Data")
        all_data = get_all_data()
        csv_data = all_data.to_csv(index=False)

        # Save CSV data to a fil
        with open("patients_data.csv", "w") as file:
            file.write(csv_data)
        
        # Create a password-protected ZIP file
        zip_file = "encrypted_patients_data.zip"
        password = "12345"
        pyminizip.compress("patients_data.csv", None, zip_file, password, 5)

        # Read the ZIP file
        with open(zip_file, "rb") as file:
            zip_data = file.read()

        # Allow users to download the password-protected ZIP file
        st.download_button(
            label="Download Encrypted ZIP",
            data=zip_data,
            file_name=zip_file,
            mime="application/zip"
        )

    st.image('divider.png')




    st.image('divider.png')

    st.subheader(':blue[Instructions to Nurse]')
    st.write('1. Add new patients by selecting "New Patient" from the dropdown menu and if it is a old patient and wants all info please click existing user and retrieve his details')
    st.write('2. There are 4 tests being done. Each has a audio guide choose your language and make sure you do not have any confusion before starting the process')
    st.write('3. Monitor patient carefully and any adverse behaviour reach the medical team immediately')
    st.write('4. Incase of any doubts contact us - bytebuddies@gmail.com')
    st.write('')
    st.write('')
    st.write('')
    st.write('')


if __name__ == "_main_":
    app()