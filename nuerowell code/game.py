import math
import random
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import cvzone
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Google Sheets API credentials....
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("neurowell-5b8eaaee5d15.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1lRSSzi6IfIEEKkfpMj67QI3S4ukyKzr2_yDlVEFGQcc/edit?gid=0#gid=0").sheet1

# Initializing hand detector..
detector = HandDetector(detectionCon=0.8, maxHands=1)

class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []
        self.lengths = []  
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = (0, 0)
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        if self.imgFood is None:
            raise ValueError(f"Could not read image at {pathFood}")
        if self.imgFood.shape[2] == 3:  
            self.imgFood = cv2.cvtColor(self.imgFood, cv2.COLOR_BGR2BGRA)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = (0, 0)
        self.score = 0
        self.gameOver = False
        self.randomFoodLocation()
        
    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)
        
    def update(self, imgMain, currentHead):
        if self.gameOver:
            st.error("Game Over")
            st.write(f'Your Score: {self.score}')
            return imgMain

        px, py = self.previousHead
        cx, cy = currentHead

        self.points.append([cx, cy])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        # Length Reduction
        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths.pop(i)
                self.points.pop(i)
                if self.currentLength < self.allowedLength:
                    break
       
        # for  Drawing Snake..
        if self.points:
            for i, point in enumerate(self.points):
                if i != 0:
                    cv2.line(imgMain, tuple(self.points[i - 1]), tuple(self.points[i]), (0, 0, 255), 20)
            cv2.circle(imgMain, tuple(self.points[-1]), 20, (0, 255, 0), cv2.FILLED)

        # for Drawing Food...
        rx, ry = self.foodPoint
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

       
        if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
                ry - self.hFood // 2 < cy < ry + self.hFood // 2:
            self.randomFoodLocation()
            self.allowedLength += 50
            self.score += 1
            st.success(f"Score: {self.score}")

        return imgMain
    
if 'progress' not in st.session_state:
        st.session_state.progress = []

    # Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

game = SnakeGameClass("images/Donut.png")

def app():
    st.title("Object recognising and  Hand Tracking Test Analysis")
    st.image('divider.png')

    st.write(" - This game evaluates the patient’s concentration and eye-hand coordination by requiring them to navigate a snake through the obstacles.")
    st.write(" - :orange[Purpose of test]: It helps in assessing and enhancing the patient’s focus, reaction time, and precision, which are essential for daily activities and overall cognitive rehabilitation.")

    # Initialize session state
    

    with st.form("game_form"):

        st.image('images/sukhaa.jpg')
        st.write('')
        start_button = st.form_submit_button("Click here to start the test analysis")

        if start_button:
            FRAME_WINDOW = st.image([])
            start_time = time.time()
            end_time = start_time + 40  # Run the game for 40 seconds

            while time.time() < end_time:
                success, img = cap.read()
                if not success:
                    break
                img = cv2.flip(img, 1)
                hands, img = detector.findHands(img, flipType=False)

                if hands:
                    lmList = hands[0]['lmList']
                    pointIndex = lmList[8][0:2]
                    img = game.update(img, pointIndex)

                remaining_time = int(end_time - time.time())
                cv2.putText(img, f"Time Left: {remaining_time}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

                FRAME_WINDOW.image(img, channels="BGR", use_column_width=True)

            cap.release()
            cv2.destroyAllWindows()

            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            st.session_state.progress.append({
                'timestamp': timestamp,
                'time': 40,
                'score': game.score,
            })

            if st.session_state.progress:
                df = pd.DataFrame(st.session_state.progress)
                st.write("Game Recognition Progress")
                st.dataframe(df)

    patient_name = st.text_input("Enter patient name:")

        
    score = st.number_input("Enter the score patient made in 40 seconds:", min_value=0)
    

        # Button to calculate and update the  score....
    if st.button("Upload Score"):
            # Calculate the  score...
            speech_score = score

            # Search for the patient by name and update the score...
            try:
                cell = sheet.find(patient_name)
                row_index = cell.row
                sheet.update_cell(row_index, 6, speech_score)
                st.success(f"Snake score updated successfully for {patient_name} in Google Sheets!")
            except gspread.exceptions.CellNotFound:
                st.error("Patient not found. Please check the name and try again.")

    
    

# Run the app
if __name__ == "__main__":
    app()