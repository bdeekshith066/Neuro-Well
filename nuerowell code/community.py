# The Community page showcases real-life stories from stroke survivors and caregivers, fostering support, motivation, and knowledge sharing. Members share recovery journeys, 
# challenges, and rehabilitation strategies, creating a compassionate space for encouragement and collective growth.


import streamlit as st
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
        font-size: 3.2em;
    }
    </style>
    <div class="gradient-text">Community</div>
    """
    # Render the gradient text
    st.markdown(gradient_text_html, unsafe_allow_html=True)
    st.write(" :orange[Where Strength Meets Support: Join Our Community of Stroke Warriors and Caregivers]")

    st.image('images/divider.png')

    st.write("Our community page  serves as a supportive hub where stroke survivors and caregivers come together to share experiences, seek advice, and find inspiration. It's a place where members celebrate milestones, offer encouragement during challenges, and exchange practical tips for rehabilitation. Through heartfelt stories and shared journeys, our community fosters a sense of belonging and empowerment, promoting emotional resilience and motivation. ")
    col1, col2, col3, col4, col5 = st.columns([1,0.15,1,0.15,1])

    with col1:
        with st.form("person1"):
            st.write("")
            st.image('images/person1.jpg')
            st.write('ohns Recovery Journey: John, a stroke survivor, shares his progress and challenges on the community page. He discusses how he regained mobility through consistent physiotherapy exercises recommended by Neuro Well. Other patients find motivation in Johns journey and share their own recovery milestones, fostering a supportive environment.')
            st.form_submit_button("Thanks to NeuroWell ")

    with col3:
        with st.form("person2"):
            st.write("")
            st.image('images/person2.jpg')
            st.write('Emmas Emotional Support: Emma, newly diagnosed with stroke-related aphasia, uses the community page to connect with others facing similar communication challenges. Through shared experiences and tips from fellow members, Emma learns new strategies to improve her speech and feels less isolated during her recovery.')
            st.form_submit_button("NeuroWell Helped Me")

    with col5:
        with st.form("person3"):
            st.write("")
            st.image('images/person3.jpg')
            st.write('Davids Rehabilitation Insights: David, who benefited from using an exoskeleton recommended by Neuro Well, posts about his experience with the device on the community page. His detailed feedback helps other patients and caregivers understand how exoskeletons can aid in rehabilitation, sparking discussions and questions that deepen everyones understanding.')    
            st.form_submit_button("Thanks to daily Physio")
