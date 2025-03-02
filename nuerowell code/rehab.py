import cv2   # Handles video capture, image processing, and drawing text on frames.
import mediapipe as mp # Provides pre-trained models for hand landmark detection.
import streamlit as st

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands    #provides a pre-trained hand tracking model
mp_drawing = mp.solutions.drawing_utils 
#mp_drawing (from mediapipe.solutions.drawing_utils) is used to draw hand landmarks and connections on the image.
#This makes it easier to visualize the detected hand and its key points.

# Function to count fingers
#Compares the Y-coordinates of each fingertip with its proximal interphalangeal (PIP) joint.
#If a fingertip is higher than its PIP, it is considered raised.
#Thumb: Uses x-coordinates because it moves sideways.

def count_fingers(hand_landmarks):
    if hand_landmarks:
        fingers = [
            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
        ]
        
        finger_count = sum(fingers)
        
        return finger_count
    return 0


# Function to check if hand is oriented correctly
#It prevents errors in finger detection by filtering out upside-down hands.
def is_hand_oriented_correctly(hand_landmarks):
     # Get the Y-coordinate of the wrist
    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    # Get the Y-coordinate of the middle finger tip
    middle_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    # The hand is considered "correctly oriented" if the middle finger tip is above the wrist
    return middle_finger_tip_y < wrist_y


# Function to run the finger counting detection and store metrics
def run_finger_detection():
    cap = cv2.VideoCapture(0)
    right_hand_done = False
    left_hand_done = False

    FRAME_WINDOW = st.image([])

    # Metrics storage
    metrics = {
        "right_hand": [],
        "left_hand": []
    }

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture image from camera.")
                break
                
            #Converts frame from BGR to RGB (since OpenCV uses BGR but MediaPipe uses RGB).
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Processes the image using MediaPipe Hands (hands.process(image)).
            results = hands.process(image)
            #Converts the frame back to BGR for OpenCV display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    if is_hand_oriented_correctly(hand_landmarks):
                        finger_count = count_fingers(hand_landmarks)
                        if not right_hand_done and hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                            cv2.putText(image, f'Right Hand Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            if finger_count == 5:
                                right_hand_done = True
                                cv2.putText(image, 'Good job! Right hand finger counting complete.', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                FRAME_WINDOW.image(image, channels="BGR")
                                st.write("Good job! Right hand finger counting complete.")
                                st.write("Please show your left hand for finger counting.")
                                cv2.waitKey(2000)
                                metrics["right_hand"].append(finger_count)

                        if right_hand_done and not left_hand_done and hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x > 0.5:
                            cv2.putText(image, f'Left Hand Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            if finger_count == 5:
                                left_hand_done = True
                                cv2.putText(image, 'Good job! Left hand finger counting complete.', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                FRAME_WINDOW.image(image, channels="BGR")
                                st.write("Good job! Left hand finger counting complete.")
                                metrics["left_hand"].append(finger_count)
                                cap.release()
                                cv2.destroyAllWindows()
                                return metrics
                    else:
                        cv2.putText(image, 'Orient your hand upright', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            FRAME_WINDOW.image(image, channels="BGR")

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return metrics

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define state machine states
STATE_LEFT_WRIST = 0 #Detect left-hand wrist movement. Track the Y-coordinate of the left wrist. Compare it with the previous frame’s Y-coordinate to detect movement.
STATE_RIGHT_WRIST = 1
STATE_LEFT_FINGER = 2
STATE_RIGHT_FINGER = 3

# Initialize state
state = STATE_LEFT_WRIST

# Initialize counters and thresholds
prev_wrist_y_left = None   # Stores previous Y-position of the left wrist
prev_wrist_y_right = None
wrist_movement_count_left = 0
wrist_movement_count_right = 0

movement_threshold = 20   #If the wrist moves by at least 20 pixels (or units), it counts as one movement
rounds_left_hand = 0
rounds_right_hand = 0
rounds_threshold = 5
expected_finger_count = 1

# Define the function to count fingers
def count_fingers(hand_landmarks):
    if not hand_landmarks:
        return 0
    
    thumb_is_open = hand_landmarks[4].x > hand_landmarks[3].x

    fingers = []
    for tip, base in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(hand_landmarks[tip].y < hand_landmarks[base].y)

    fingers_open = [thumb_is_open] + fingers
    count = sum(fingers_open)
    
    return count

# Streamlit app layout configuration
def app():
    global state, prev_wrist_y_left, prev_wrist_y_right, wrist_movement_count_left, wrist_movement_count_right
    global rounds_left_hand, rounds_right_hand

    st.title('Wrist and Finger Tracking with MediaPipe and Streamlit')

    # Start button to initiate video capture
    if st.button('Start Tracking'):
        cap = cv2.VideoCapture(0)

        FRAME_WINDOW = st.image([])

        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to capture image from camera.")
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                                  mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

                # State Machine Logic
                if state == STATE_LEFT_WRIST:
                    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                        hand = results.multi_hand_landmarks[0]
                        wrist_y = hand.landmark[0].y * image.shape[0]
                        if prev_wrist_y_left is not None and abs(wrist_y - prev_wrist_y_left) > movement_threshold:
                            wrist_movement_count_left += 1
                            rounds_left_hand += 1
                        prev_wrist_y_left = wrist_y
                    cv2.putText(image, f'Left Wrist Movements: {wrist_movement_count_left}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    if rounds_left_hand >= rounds_threshold:
                        state = STATE_RIGHT_WRIST
                        rounds_left_hand = 0
                        wrist_movement_count_left = 0
                        cv2.putText(image, 'OK, great! Switch to right hand!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                elif state == STATE_RIGHT_WRIST:
                    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
                        hand = results.multi_hand_landmarks[1]
                        wrist_y = hand.landmark[0].y * image.shape[0]
                        if prev_wrist_y_right is not None and abs(wrist_y - prev_wrist_y_right) > movement_threshold:
                            wrist_movement_count_right += 1
                            rounds_right_hand += 1
                        prev_wrist_y_right = wrist_y
                    cv2.putText(image, f'Right Wrist Movements: {wrist_movement_count_right}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    if rounds_right_hand >= rounds_threshold:
                        state = STATE_LEFT_FINGER
                        rounds_right_hand = 0
                        wrist_movement_count_right = 0
                        cv2.putText(image, 'OK, great! Show 1 finger with left hand!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                #Counts wrist movement if the wrist has moved significantly (> movement_threshold).
                elif state == STATE_LEFT_FINGER:
                    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        finger_count = count_fingers(hand_landmarks.landmark)
                        cv2.putText(image, f'Left Hand Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        if finger_count == expected_finger_count:
                            state = STATE_RIGHT_FINGER
                            cv2.putText(image, 'Great! Now show 1 finger with right hand!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                elif state == STATE_RIGHT_FINGER:
                    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
                        hand_landmarks = results.multi_hand_landmarks[1]
                        finger_count = count_fingers(hand_landmarks.landmark)
                        cv2.putText(image, f'Right Hand Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        if finger_count == expected_finger_count:
                            cv2.putText(image, 'Great! Task completed!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            state = STATE_LEFT_WRIST

                FRAME_WINDOW.image(image, channels="BGR")

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        
        # Releases the webcam and closes all OpenCV windows.
        cap.release()
        cv2.destroyAllWindows()

    st.write('')
    st.write('')

if __name__ == "__main__":
    app()
