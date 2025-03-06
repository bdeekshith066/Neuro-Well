import cv2
import numpy as np
import pandas as pd
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import cvzone
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

scoree = [0, 0]

# Set up Google Sheets API credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("neurowell-5b8eaaee5d15.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1lRSSzi6IfIEEKkfpMj67QI3S4ukyKzr2_yDlVEFGQcc/edit?gid=0#gid=0").sheet1

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Initialize game assets
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Define image paths
image_paths = {
    "background": "images/Background.jpg",
    "game_over": "images/game_over.jpg",
    "ball": "images/ddd.jpg",
    "left_bat": "images/left.jpg",
    "right_bat": "images/hight.jpg"
}

# Load and check images
images = {}
for name, path in image_paths.items():
    if os.path.exists(path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            st.error(f"Failed to load image: {path}")
            st.stop()
        # Check if the image has an alpha channel
        if image.shape[2] == 3:
            # Add an alpha channel
            alpha_channel = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
            #  Ensuring Images Have an Alpha Channel as Some images might have only three channels (BGR) instead of four (BGRA). This block ensures every image has an alpha channel (for transparency).
            # An alpha channel controls opacity, which is crucial for overlaying game elements smoothly.

            image = np.concatenate((image, alpha_channel), axis=2)
        images[name] = image
    else:
        st.error(f"Image not found: {path}")
        st.stop()

# Resize imgBackground to match the webcam frame size
imgBackground = cv2.resize(images["background"], (1280, 720))

# Ensure imgBackground has 3 channels for blending
if imgBackground.shape[2] == 4:
    imgBackground = cv2.cvtColor(imgBackground, cv2.COLOR_BGRA2BGR)
    # If the background has an alpha channel, it's converted to BGR. This prevents errors when blending with OpenCV functions that expect three-channel images.

# Initialize game variables
ballPos = [100, 100]     
# the ball starts at (100,100) in the frame.
speedX = 23
speedY = 23
# The ball moves 23 pixels per frame in both X and Y directions.
gameOver = False
#  The game starts in an active state.

def update_game():
    global ballPos, speedX, speedY, gameOver, scoree, images

    success, img = cap.read()
    if not success:
        st.error("Failed to read from webcam.")
        st.stop()

    img = cv2.flip(img, 1)  
    #  Mirrors the frame horizontally so that movement feels natural (like a mirror reflection).
    imgRaw = img.copy()
    hands, img = detector.findHands(img, flipType=False)

    # Ensure img and imgBackground have the same size and number of channels
    imgBackground_resized = cv2.resize(imgBackground, (img.shape[1], img.shape[0]))
    # Ensures the background image matches the webcam frame size.

    img = cv2.addWeighted(img, 0.2, imgBackground_resized, 0.8, 0)
    # 20% webcam feed (img) and 80% game background (imgBackground_resized)

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = images["left_bat"].shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, images["left_bat"], (59, y1))
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    scoree[0] += 1
                # If the detected left hand is near X = 59, the left bat image is overlaid.
                # If the ball overlaps the bat's position, speedX is inverted (ball bounces back).

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, images["right_bat"], (1195, y1))
                if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30
                    scoree[1] += 1
                # The right bat is placed at X = 1195.

    # Game Over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True
    # If the ball crosses X < 40 (left edge) or X > 1200 (right edge), the game ends.

    if gameOver:
        img = images["game_over"]
        cv2.putText(img, str(scoree[1] + scoree[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX,
                    2.5, (200, 0, 200), 5)
        
    else:
        # Move the Ball
        if ballPos[1] >= 500 or ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # If the ball reaches Y = 500 (bottom) or Y = 10 (top), it bounces back by inverting speedY. 
        # ballPos[0] += speedX, ballPos[1] += speedY → Moves the ball by updating its position every frame.

        # Draw the ball
        img = cvzone.overlayPNG(img, images["ball"], ballPos)

        cv2.putText(img, str(scoree[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 5)
        cv2.putText(img, str(scoree[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 5)

    # Ensure imgRaw has 3 channels
    if imgRaw.shape[2] == 4:
        imgRaw = cv2.cvtColor(imgRaw, cv2.COLOR_BGRA2BGR)

    return img

def app():
    
    st.title("Eye–hand coordination Test Analysis")
    st.image('divider.png')

    st.write(" - This computer vision-based game assesses the coordination between a patient’s hands, reflecting the functional status of both the left and right hemispheres of the brain.")
    st.write(" - :orange[Purpose of test]: It provides insights into motor skills and bilateral coordination, crucial for designing targeted rehabilitation strategies.")
    # Initialize session state
    if 'progress' not in st.session_state:
        st.session_state.progress = []

    with st.form("game_form"):

        st.image('coordination.jpg')
        st.write('')
        start_button = st.form_submit_button("Click here to start the test analysis")

        if start_button:

            cap = cv2.VideoCapture(0)
            cap.set(3, 1280)
            cap.set(4, 720)
            FRAME_WINDOW = st.image([])

            start_time = time.time()
            end_time = start_time + 40  # Run the game for 40 seconds

            while time.time() < end_time:
                img = update_game()
                FRAME_WINDOW.image(img, channels="BGR", use_column_width=True)

                remaining_time = int(end_time - time.time())
                cv2.putText(img, f"Time Left: {remaining_time}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

                if gameOver:
                    break

            cap.release()
            cv2.destroyAllWindows()

            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            st.session_state.progress.append({
                'timestamp': timestamp,
                'time': 40,
                'scoree': scoree[1] + scoree[0],
            })

            if st.session_state.progress:
                df = pd.DataFrame(st.session_state.progress)
                st.write("Game Recognition Progress")
                st.dataframe(df)

    patient_name = st.text_input("Enter patient name:")

    # Input fields for matched words and total words
    patient_scoree = st.number_input("Enter the score patient made in 40 seconds:", min_value=0)

    # Button to calculate and update the speech score
    if st.button("Upload Score"):
        # Calculate the speech score
        speech_scoree = patient_scoree

        # Search for the patient by name and update the score
        try:
            cell = sheet.find(patient_name)
            row_index = cell.row
            sheet.update_cell(row_index, 7, speech_scoree)
            st.success(f"Speech score updated successfully for {patient_name} in Google Sheets!")
        except gspread.exceptions.CellNotFound:
            st.error("Patient not found. Please check the name and try again.")

    st.write('') 
    st.write('')  

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        st.error(f"An error occurred: {e}")
