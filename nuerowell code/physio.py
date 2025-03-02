import streamlit as st
import speech_recognition as sr
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import base64
import json
from PIL import Image
from streamlit_autorefresh import st_autorefresh
import time as tm
from gtts import gTTS
from googletrans import Translator
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Set up Google Sheets API credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("neurowell-5b8eaaee5d15.json", scope)
client = gspread.authorize(creds)
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1lRSSzi6IfIEEKkfpMj67QI3S4ukyKzr2_yDlVEFGQcc/edit?gid=0#gid=0").sheet1


# Function to recognize speech using the microphone
def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        response = recognizer.recognize_google(audio)
    except sr.RequestError:
        response = "API unavailable"
    except sr.UnknownValueError:
        response = "Unable to recognize speech"
    return response

# Initialize recognizer and microphone
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Sample paragraphs
paragraphs = [
    "Exercising daily is good for your body and mind. Simple activities like walking or stretching can boost your mood. Exercise can also help improve your strength and balance.",
    
]

# Initialize session state for storing progress
if 'progress' not in st.session_state:
    st.session_state.progress = []

# Initialize session state for paragraph selection
if 'paragraph' not in st.session_state:
    st.session_state.paragraph = random.choice(paragraphs)


translator = Translator()

# Define the feedback text
feedback_text = "Ask the patient to speak which is written below Please read the following paragraph aloud. Observe him carefully while they speak and the scores will be displayed on the screen. Assess him with care regards NueroWell"
feedback_textt = "Ask the patient to clearly observe the image and match him with the grid od emojis which are displayed below.Each will be displayed for 115 seconds and a correct one will have +9 score and wrong will have -1 seconds and Assess him with care regards Team NueroWell "
    # List of languages
languages = {
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn",
        "Telugu": "te",
        "Tamil": "ta",
        "Malayalam": "ml",
        "Gujarati": "gu",
        "Marathi": "mr",
        "Bengali": "bn",
        "Punjabi": "pa"
    }

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
          <div class="gradient-text">Neuro Well Test Analysis</div>
          """

      # Render the gradient text
    st.markdown(gradient_text_html, unsafe_allow_html=True)
    
    st.image('divider.png')


    

    # Create columns
    col11, col22 = st.columns(2)

    # Language selection in column 2
    with col22:
        selected_language = st.selectbox("Choose a language for instructions", list(languages.keys()))

    # Translate the text
    if selected_language != "English":
        translated_text = translator.translate(feedback_text, dest=languages[selected_language]).text
    else:
        translated_text = feedback_text

    # Convert text to speech
    tts = gTTS(translated_text, lang=languages[selected_language])
    audio_file = "feedback.mp3"
    tts.save(audio_file)

    # Display the audio in column 1
    with col11:
        st.write('')
        st.audio(audio_file)

    # Remove the audio file after playing
    

    col8, col9 = st.columns([3, 1.5])

    with col8:

        st.subheader(" 1. :orange[Speech Test Analysis] ")
        
        with st.form("speech"):
            # Display the paragraph
            st.write("Please read the following paragraph aloud:")
            st.write(st.session_state.paragraph)

            # Button to start speech recognition
            if st.form_submit_button('Start Speech Recognition'):
                with st.spinner("Listening..."):
                    response = recognize_speech_from_mic(recognizer, microphone)
                st.write(f"Recognized Speech: {response}")

                # Get the current timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Calculate score based on the recognized speech
                original_words = set(st.session_state.paragraph.split())
                recognized_words = set(response.split())
                matched_words = original_words.intersection(recognized_words)
                score = len(matched_words)

                # Store the response and score in session state
                st.session_state.progress.append({
                    'timestamp': timestamp,
                    'response': response,
                    'score': score,
                    'total_words': len(original_words),
                    'matched_words': len(matched_words),
                    'unmatched_words': len(original_words) - len(matched_words),
                    
                })

                # Save progress to a CSV file (or database)
                df = pd.DataFrame(st.session_state.progress)
                df.to_csv("progress.csv", index=False)

            # Display progress as a table
            if st.session_state.progress:
                df = pd.DataFrame(st.session_state.progress)
                st.write("Speech Recognition Progress")
                st.dataframe(df)
        # Calculate the cumulative score to be updated in Google Sheets
        

        patient_name = st.text_input("Enter patient name:")

        # Input fields for matched words and total words
        matched_words = st.number_input("Enter the number of matched words:", min_value=0)
        total_words = st.number_input("Enter the total number of words:", min_value=1)

        # Button to calculate and update the speech score
        if st.button("Upload Score"):
            # Calculate the speech score
            speech_score = (matched_words / total_words) * 10

            # Search for the patient by name and update the score
            try:
                cell = sheet.find(patient_name)
                row_index = cell.row
                sheet.update_cell(row_index, 4, speech_score)
                st.success(f"Speech score updated successfully for {patient_name} in Google Sheets!")
            except gspread.exceptions.CellNotFound:
                st.error("Patient not found. Please check the name and try again.")
                

    with col9:
        # Check if there is progress data to display graphs
        if st.session_state.progress:
            df = pd.DataFrame(st.session_state.progress)

            # Select box for graph type
            graph_choice = st.selectbox("Select Graph", ["Pie Chart", "Stacked Bar Chart", "Area Chart"])

            # Display the selected graph
            if graph_choice == "Area Chart":
                st.write("Progress Over Time (Area Chart)")
                plt.figure(figsize=(8, 4))
                plt.fill_between(df['timestamp'], df['score'], color="skyblue", alpha=0.4)
                plt.plot(df['timestamp'], df['score'], color="Slateblue", alpha=0.6)
                plt.xlabel('Timestamp')
                plt.ylabel('Score')
                plt.title('Speech Recognition Progress')
                plt.xticks(rotation=0)
                plt.grid(True)
                st.pyplot(plt)


            elif graph_choice == "Stacked Bar Chart":
                st.write("Scores (Stacked Bar Chart)")
                df.set_index('timestamp', inplace=True)
                df[['matched_words', 'unmatched_words']].plot(kind='bar', stacked=True, figsize=(8, 4), color=['#66b3ff', '#ff9999'])
                plt.xlabel('Timestamp')
                plt.ylabel('Count')
                plt.title('Matched vs Unmatched Words')
                plt.xticks(rotation=0)
                plt.grid(axis='y')
                st.pyplot(plt)

            elif graph_choice == "Pie Chart":
                st.write("Word Recognition (Pie Chart)")
                total_words = df['total_words'].sum()
                matched_words = df['matched_words'].sum()
                unmatched_words = total_words - matched_words
                labels = ['Matched Words', 'Unmatched Words']
                sizes = [matched_words, unmatched_words]
                colors = ['#ff9999', '#66b3ff']
                explode = (0.1, 0)  # explode 1st slice
                plt.figure(figsize=(8, 4))
                plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                        shadow=True, startangle=140)
                plt.title('Matched vs Unmatched Words')
                st.pyplot(plt)

            st.write('')

    st.image('divider.png')
    st.image('divider.png')
    
    col111, col222 = st.columns(2)

    # Language selection in column 2
    with col222:
        selected_language = st.selectbox("Choose a language for  the instructions", list(languages.keys()))

    # Translate the text
    if selected_language != "English":
        translated_text = translator.translate(feedback_textt, dest=languages[selected_language]).text
    else:
        translated_text = feedback_textt

    # Convert text to speech
    tts = gTTS(translated_text, lang=languages[selected_language])
    audio_file = "feedbackk.mp3"
    tts.save(audio_file)

    # Display the audio in column 1
    with col111:
        st.write('')
        st.audio(audio_file)

    # Remove the audio file after playing
    os.remove(audio_file)
    
    st.subheader(" 2. :orange[PicMatch Test Analysis]")
    with st.form("game"):
        vDrive = os.path.splitdrive(os.getcwd())[0]
        if vDrive == "C:": vpth = "C:/Users/Shawn/dev/utils/pixmatch/"   # local developer's disc
        else: vpth = "./"

        sbe = """<span style='font-size: 140px;
                            border-radius: 7px;
                            text-align: center;
                            display:inline;
                            padding-top: 3px;
                            padding-bottom: 3px;
                            padding-left: 0.4em;
                            padding-right: 0.4em;
                            '>
                            |fill_variable|
                            </span>"""

        pressed_emoji = """<span style='font-size: 24px;
                                        border-radius: 7px;
                                        text-align: center;
                                        display:inline;
                                        padding-top: 3px;
                                        padding-bottom: 3px;
                                        padding-left: 0.2em;
                                        padding-right: 0.2em;
                                        '>
                                        |fill_variable|
                                        </span>"""

        horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #635985;'><br>"    # thin divider line
        purple_btn_colour = """
                                <style>
                                    div.stButton > button:first-child {background-color: #4b0082; color:#ffffff;}
                                    div.stButton > button:hover {background-color: RGB(0,112,192); color:#ffffff;}
                                    div.stButton > button:focus {background-color: RGB(47,117,181); color:#ffffff;}
                                </style>
                            """

        mystate = st.session_state
        if "expired_cells" not in mystate: mystate.expired_cells = []
        if "myscore" not in mystate: mystate.myscore = 0
        if "plyrbtns" not in mystate: mystate.plyrbtns = {}
        if "sidebar_emoji" not in mystate: mystate.sidebar_emoji = ''
        if "emoji_bank" not in mystate: mystate.emoji_bank = []
        if "GameDetails" not in mystate: mystate.GameDetails = ['Medium', 6, 7, '']  # difficulty level, sec interval for autogen, total_cells_per_row_or_col, player name

        # common functions
        def ReduceGapFromPageTop(wch_section = 'main page'):
            if wch_section == 'main page': st.markdown(" <style> div[class^='block-container'] { padding-top: 2rem; } </style> ", True) # main area
            elif wch_section == 'sidebar': st.markdown(" <style> div[class^='st-emotion-cache-10oheav'] { padding-top: 0rem; } </style> ", True) # sidebar
            elif wch_section == 'all': 
                st.markdown(" <style> div[class^='block-container'] { padding-top: 2rem; } </style> ", True) # main area
                st.markdown(" <style> div[class^='st-emotion-cache-10oheav'] { padding-top: 0rem; } </style> ", True) # sidebar            

        def ReadPictureFile(wch_fl):
            try:
                pxfl = f"{vpth}{wch_fl}"
                return base64.b64encode(open(pxfl, 'rb').read()).decode()

            except: return ""

        def PressedCheck(vcell):
            if mystate.plyrbtns[vcell]['isPressed'] == False:
                mystate.plyrbtns[vcell]['isPressed'] = True
                mystate.expired_cells.append(vcell)

                if mystate.plyrbtns[vcell]['eMoji'] == mystate.sidebar_emoji:
                    mystate.plyrbtns[vcell]['isTrueFalse'] = True
                    mystate.myscore += 5

                    if mystate.GameDetails[0] == 'Easy': mystate.myscore += 5
                    elif mystate.GameDetails[0] == 'Medium': mystate.myscore += 3
                    elif mystate.GameDetails[0] == 'Hard': mystate.myscore += 1
                
                else:
                    mystate.plyrbtns[vcell]['isTrueFalse'] = False
                    mystate.myscore -= 1

        def ResetBoard():
            total_cells_per_row_or_col = mystate.GameDetails[2]

            sidebar_emoji_no = random.randint(1, len(mystate.emoji_bank))-1
            mystate.sidebar_emoji = mystate.emoji_bank[sidebar_emoji_no]

            sidebar_emoji_in_list = False
            for vcell in range(1, ((total_cells_per_row_or_col ** 2)+1)):
                rndm_no = random.randint(1, len(mystate.emoji_bank))-1
                if mystate.plyrbtns[vcell]['isPressed'] == False:
                    vemoji = mystate.emoji_bank[rndm_no]
                    mystate.plyrbtns[vcell]['eMoji'] = vemoji
                    if vemoji == mystate.sidebar_emoji: sidebar_emoji_in_list = True

            if sidebar_emoji_in_list == False:  # sidebar pix is not on any button; add pix randomly
                tlst = [x for x in range(1, ((total_cells_per_row_or_col ** 2)+1))]
                flst = [x for x in tlst if x not in mystate.expired_cells]
                if len(flst) > 0:
                    lptr = random.randint(0, (len(flst)-1))
                    lptr = flst[lptr]
                    mystate.plyrbtns[lptr]['eMoji'] = mystate.sidebar_emoji

        def PreNewGame():
            total_cells_per_row_or_col = mystate.GameDetails[2]
            mystate.expired_cells = []
            mystate.myscore = 0

            
        
            foods = ['🍏', '🍎', '🍐', '🍊', '🍋', '🍌', '🍉', '🍇', '🍓', '🍈', '🍒', '🍑', '🥭', '🍍', '🥥', '🥝', '🍅', '🍆', '🥑', '🥦', '🥬', '🥒', '🌽', '🥕', '🧄', '🧅', '🥔', '🍠', '🥐', '🥯', '🍞', '🥖', '🥨', '🧀', '🥚', '🍳', '🧈', '🥞', '🧇', '🥓', '🥩', '🍗', '🍖', '🦴', '🌭', '🍔', '🍟', '🍕']
            clocks = ['🕓', '🕒', '🕑', '🕘', '🕛', '🕚', '🕖', '🕙', '🕔', '🕤', '🕠', '🕕', '🕣', '🕞', '🕟', '🕜', '🕢', '🕦']
            hands = ['🤚', '🖐', '✋', '🖖', '👌', '🤏', '✌️', '🤞', '🤟', '🤘', '🤙', '👈', '👉', '👆', '🖕', '👇', '☝️', '👍', '👎', '✊', '👊', '🤛', '🤜', '👏', '🙌', '🤲', '🤝', '🤚🏻', '🖐🏻', '✋🏻', '🖖🏻', '👌🏻', '🤏🏻', '✌🏻', '🤞🏻', '🤟🏻', '🤘🏻', '🤙🏻', '👈🏻', '👉🏻', '👆🏻', '🖕🏻', '👇🏻', '☝🏻', '👍🏻', '👎🏻', '✊🏻', '👊🏻', '🤛🏻', '🤜🏻', '👏🏻', '🙌🏻', '🤚🏽', '🖐🏽', '✋🏽', '🖖🏽', '👌🏽', '🤏🏽', '✌🏽', '🤞🏽', '🤟🏽', '🤘🏽', '🤙🏽', '👈🏽', '👉🏽', '👆🏽', '🖕🏽', '👇🏽', '☝🏽', '👍🏽', '👎🏽', '✊🏽', '👊🏽', '🤛🏽', '🤜🏽', '👏🏽', '🙌🏽']
            animals = ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼', '🐨', '🐯', '🦁', '🐮', '🐷', '🐽', '🐸', '🐵', '🙈', '🙉', '🙊', '🐒', '🐔', '🐧', '🐦', '🐤', '🐣', '🐥', '🦆', '🦅', '🦉', '🦇', '🐺', '🐗', '🐴', '🦄', '🐝', '🐛', '🦋', '🐌', '🐞', '🐜', '🦟', '🦗', '🦂', '🐢', '🐍', '🦎', '🦖', '🦕', '🐙', '🦑', '🦐', '🦞', '🦀', '🐡', '🐠', '🐟', '🐬', '🐳', '🐋', '🦈', '🐊', '🐅', '🐆', '🦓', '🦍', '🦧', '🐘', '🦛', '🦏', '🐪', '🐫', '🦒', '🦘', '🐃', '🐂', '🐄', '🐎', '🐖', '🐏', '🐑', '🦙', '🐐', '🦌', '🐕', '🐩', '🦮', '🐕‍🦺', '🐈', '🐓', '🦃', '🦚', '🦜', '🦢', '🦩', '🐇', '🦝', '🦨', '🦦', '🦥', '🐁', '🐀', '🦔']
            vehicles = ['🚗', '🚕', '🚙', '🚌', '🚎', '🚓', '🚑', '🚒', '🚐', '🚚', '🚛', '🚜', '🦯', '🦽', '🦼', '🛴', '🚲', '🛵', '🛺', '🚔', '🚍', '🚘', '🚖', '🚡', '🚠', '🚟', '🚃', '🚋', '🚞', '🚝', '🚄', '🚅', '🚈', '🚂', '🚆', '🚇', '🚊', '🚉', '✈️', '🛫', '🛬', '💺', '🚀', '🛸', '🚁', '🛶', '⛵️', '🚤', '🛳', '⛴', '🚢']
            
            moon = ['🌕', '🌔', '🌓', '🌗', '🌒', '🌖', '🌑', '🌜', '🌛', '🌙']

            random.seed()
            if mystate.GameDetails[0] == 'Easy':
                wch_bank = random.choice(['foods'])
                mystate.emoji_bank = locals()[wch_bank]

            
            mystate.plyrbtns = {}
            for vcell in range(1, ((total_cells_per_row_or_col ** 2)+1)): mystate.plyrbtns[vcell] = {'isPressed': False, 'isTrueFalse': False, 'eMoji': ''}

        def ScoreEmoji():
            if mystate.myscore == 0: return '😐'
            elif -5 <= mystate.myscore <= -1: return '😏'
            elif -10 <= mystate.myscore <= -6: return '☹️'
            elif mystate.myscore <= -11: return '😖'
            elif 1 <= mystate.myscore <= 5: return '🙂'
            elif 6 <= mystate.myscore <= 10: return '😊'
            elif mystate.myscore > 10: return '😁'

        def NewGame():
            ResetBoard()
            total_cells_per_row_or_col = mystate.GameDetails[2]

            ReduceGapFromPageTop('sidebar')
            
            

            st.markdown(sbe.replace('|fill_variable|', mystate.sidebar_emoji), True)

            aftimer = st_autorefresh(interval=(mystate.GameDetails[1] * 1000), key="aftmr")
            if aftimer > 0: mystate.myscore -= 1

            st.info(f"{ScoreEmoji()} Score: {mystate.myscore} | Pending: {(total_cells_per_row_or_col ** 2)-len(mystate.expired_cells)}")

            st.markdown(horizontal_bar, True)
            
            
            

            # Set Board Dafaults
            st.markdown("<style> div[class^='css-1vbkxwb'] > p { font-size: 1.5rem; } </style> ", unsafe_allow_html=True)  # make button face big

            for i in range(1, (total_cells_per_row_or_col+1)):
                tlst = ([1] * total_cells_per_row_or_col) + [2] # 2 = rt side padding
                globals()['cols' + str(i)] = st.columns(tlst)
            
            for vcell in range(1, (total_cells_per_row_or_col ** 2)+1):
                if 1 <= vcell <= (total_cells_per_row_or_col * 1):
                    arr_ref = '1'
                    mval = 0

                elif ((total_cells_per_row_or_col * 1)+1) <= vcell <= (total_cells_per_row_or_col * 2):
                    arr_ref = '2'
                    mval = (total_cells_per_row_or_col * 1)

                elif ((total_cells_per_row_or_col * 2)+1) <= vcell <= (total_cells_per_row_or_col * 3):
                    arr_ref = '3'
                    mval = (total_cells_per_row_or_col * 2)

                elif ((total_cells_per_row_or_col * 3)+1) <= vcell <= (total_cells_per_row_or_col * 4):
                    arr_ref = '4'
                    mval = (total_cells_per_row_or_col * 3)

                elif ((total_cells_per_row_or_col * 4)+1) <= vcell <= (total_cells_per_row_or_col * 5):
                    arr_ref = '5'
                    mval = (total_cells_per_row_or_col * 4)

                elif ((total_cells_per_row_or_col * 5)+1) <= vcell <= (total_cells_per_row_or_col * 6):
                    arr_ref = '6'
                    mval = (total_cells_per_row_or_col * 5)

                elif ((total_cells_per_row_or_col * 6)+1) <= vcell <= (total_cells_per_row_or_col * 7):
                    arr_ref = '7'
                    mval = (total_cells_per_row_or_col * 6)

                elif ((total_cells_per_row_or_col * 7)+1) <= vcell <= (total_cells_per_row_or_col * 8):
                    arr_ref = '8'
                    mval = (total_cells_per_row_or_col * 7)

                elif ((total_cells_per_row_or_col * 8)+1) <= vcell <= (total_cells_per_row_or_col * 9):
                    arr_ref = '9'
                    mval = (total_cells_per_row_or_col * 8)

                elif ((total_cells_per_row_or_col * 9)+1) <= vcell <= (total_cells_per_row_or_col * 10):
                    arr_ref = '10'
                    mval = (total_cells_per_row_or_col * 9)
                    
                globals()['cols' + arr_ref][vcell-mval] = globals()['cols' + arr_ref][vcell-mval].empty()
                if mystate.plyrbtns[vcell]['isPressed'] == True:
                    if mystate.plyrbtns[vcell]['isTrueFalse'] == True:
                        globals()['cols' + arr_ref][vcell-mval].markdown(pressed_emoji.replace('|fill_variable|', '✅️'), True)
                    
                    elif mystate.plyrbtns[vcell]['isTrueFalse'] == False:
                        globals()['cols' + arr_ref][vcell-mval].markdown(pressed_emoji.replace('|fill_variable|', '❌'), True)

                else:
                    vemoji = mystate.plyrbtns[vcell]['eMoji']
                    globals()['cols' + arr_ref][vcell-mval].button(vemoji, on_click=PressedCheck, args=(vcell, ), key=f"B{vcell}")

            st.caption('') # vertical filler
            st.markdown(horizontal_bar, True)

            if len(mystate.expired_cells) == (total_cells_per_row_or_col ** 2):
                

                if mystate.myscore > 0: st.balloons()
                elif mystate.myscore <= 0: st.snow()

                tm.sleep(5)
                mystate.runpage = Main
                st.rerun()

        def Main():
            st.markdown('<style>[data-testid="stSidebar"] > div:first-child {width: 310px;}</style>', unsafe_allow_html=True,)  # reduce sidebar width
            st.markdown(purple_btn_colour, unsafe_allow_html=True)

            
            
        mystate.GameDetails[0] = 'Easy'

        
        st.write('PicMatch - Match emojis scattered across the grid to the ones displayed in the image')

        
        st.write(" :orange[Why play this game - Engage in a dynamic blend of memory exercises, problem-solving tasks, and attention training in this immersive and challenging game.]")
                

        if st.form_submit_button(f"🕹️ Start the  Game", use_container_width=True):

            if mystate.GameDetails[0] == 'Easy':
                mystate.GameDetails[1] = 20         # secs interval
                mystate.GameDetails[2] = 6         # total_cells_per_row_or_col
                    
                    
                    

                PreNewGame()
                mystate.runpage = NewGame
                st.rerun()

        st.markdown(horizontal_bar, True)

    patient_name = st.text_input("Enter the patient name:")

        # Input fields for matched words and total words
    matched_emoji = st.number_input("Enter the number of matched emoji:", min_value=0)
    total_emoji = st.number_input("Enter the total number of emoji:", min_value=1)

        # Button to calculate and update the speech score
    if st.button("Upload  the Score"):
        # Calculate the speech score
        speech_score = (matched_emoji / total_emoji) * 10

        # Search for the patient by name and update the score
        try:
            cell = sheet.find(patient_name)
            row_index = cell.row
            sheet.update_cell(row_index, 5, speech_score)
            st.success(f"Speech score updated successfully for {patient_name} in Google Sheets!")
        except gspread.exceptions.CellNotFound:
            st.error("Patient not found. Please check the name and try again.")


    if 'runpage' not in mystate: mystate.runpage = Main
    mystate.runpage()

    st.write('')

if __name__ == "__main__":
    app()
