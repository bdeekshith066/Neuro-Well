import streamlit as st
from googletrans import Translator

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
            font-size: 3em;
        }
        </style>
        <div class="gradient-text">NeuroPedia</div>
        """

    st.markdown(gradient_text_html, unsafe_allow_html=True) 
    st.write(":orange[Discover vital Neuro  information and insights tailored for all, empowering better understanding and care]") 
    st.image('divider.png')

    st.write('')

    def show_stroke_info():
        st.write(""" :orange[Stroke]  \n  \n :orange[About and Facts]  \n  A stroke, also known as a cerebrovascular accident (CVA), occurs when the blood supply to part of the brain is interrupted or reduced, preventing brain tissue from getting enough oxygen and nutrients. Within minutes, brain cells begin to die. There are two main types of strokes: ischemic, caused by a blockage in an artery, and hemorrhagic, caused by bleeding in or around the brain. Stroke is a leading cause of disability and the fifth leading cause of death worldwide. Quick medical intervention can significantly improve outcomes, making early recognition and treatment crucial.  \n  \n  :orange[Symptoms]  \n  Treatment for stroke often involves clot-busting medications and sometimes surgery to remove clots or repair damaged blood vessels. Post-stroke rehabilitation, including physical, occupational, and speech therapy, helps survivors regain lost abilities. While recovery varies, many stroke survivors make significant progress with therapy and support.  \n  \n :orange[Prevention Strategies]  \n Preventing stroke involves managing risk factors like high blood pressure, diabetes, and smoking. Adopting a healthy lifestyle, including regular exercise and a balanced diet, can reduce the likelihood of stroke. Regular check-ups and medication adherence are essential for individuals at risk. By prioritizing prevention, individuals can lower their chances of experiencing a stroke and lead healthier lives.  \n  \n
            Enter the name of a term from the HeartPedia to know more about it  """)
    # language codes based on the ISO 639-1 standard, which assigns two-letter abbreviations to each language.
    target_languages = {
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Telugu": "te",
        "Tamil": "ta",
        "Punjabi": "pa",
        "French": "fr",
        "Spanish": "es",
        "German": "de"
    }
    # Introduction
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        st.button('Stroke')
            
        st.write('')
        st.write('')
        st.write('')
        st.button('Aphasia')
        st.write('')
        st.write('')
        st.write('')
        st.button('Dysphagia:')
        st.write('')
        st.write('')
        st.write('')
        st.button('FAST')

    with col2:
        st.button('Ischemic Stroke')
        st.write('')
        st.write('')
        st.write('')
        st.button('Rehabilation')
        st.write('')
        st.write('')
        st.write('')
        st.button('Secondary Stroke Prevention')
        st.write('')
        st.write('')
        st.write('')
        st.button('Atrial Fibrillation (AFib')

    with col3:
        st.button('Hemorrhagic Stroke:')
        st.write('')
        st.write('')
        st.write('')
        st.button('Thrombolytic Therapy)')
        st.write('')
        st.write('')
        st.write('')
        st.button('Endovascular Thrombectomy')
        st.write('')
        st.write('')
        st.write('')
        st.button('Modified Rankin Scale (mRS')
    
    
    with col4:
        st.button('Transient Ischemic Attack (TIA)')
        st.write('')
        st.write('')
        st.write('')
        st.button('NIH Stroke Scale (NIHSS)')
        st.write('')
        st.write('')
        st.write('')
        st.button('Modified Rankin Scale (mRS)')
        st.write('')
        st.write('')
        st.write('')
        target_language = st.selectbox("Select Language", list(target_languages.keys()))
        
    # Language selection
    

    

    def get_pedia_info(user_input, target_language='en'):
        pedia_info = {
            "stroke": """ :orange[Stroke]  \n  \n :orange[About and Facts]  \n  A stroke, also known as a cerebrovascular accident (CVA), occurs when the blood supply to part of the brain is interrupted or reduced, preventing brain tissue from getting enough oxygen and nutrients. Within minutes, brain cells begin to die. There are two main types of strokes: ischemic, caused by a blockage in an artery, and hemorrhagic, caused by bleeding in or around the brain. Stroke is a leading cause of disability and the fifth leading cause of death worldwide. Quick medical intervention can significantly improve outcomes, making early recognition and treatment crucial.  \n  \n  :orange[Symptoms]  \n  Treatment for stroke often involves clot-busting medications and sometimes surgery to remove clots or repair damaged blood vessels. Post-stroke rehabilitation, including physical, occupational, and speech therapy, helps survivors regain lost abilities. While recovery varies, many stroke survivors make significant progress with therapy and support.  \n  \n :orange[Prevention Strategies]  \n Preventing stroke involves managing risk factors like high blood pressure, diabetes, and smoking. Adopting a healthy lifestyle, including regular exercise and a balanced diet, can reduce the likelihood of stroke. Regular check-ups and medication adherence are essential for individuals at risk. By prioritizing prevention, individuals can lower their chances of experiencing a stroke and lead healthier lives.  \n  \n
            Enter the name of a term from the HeartPedia to know more about it  """,




            "rehabilation": """ Rehabilation  \n  \n Understanding Rehabilitation  \n  Stroke rehabilitation is a crucial aspect of recovery, aiming to help survivors regain independence and improve quality of life. It encompasses various therapies, including physical, occupational, and speech therapy, tailored to individual needs. Rehabilitation efforts focus on restoring lost functions, enhancing mobility, and promoting overall well-being.   \n  \n  Treatment Approaches  \n   Rehabilitation after stroke involves a multidisciplinary approach, combining therapeutic exercises, assistive devices, and adaptive strategies. Physical therapy aims to improve strength, coordination, and mobility, while occupational therapy focuses on relearning daily activities and enhancing fine motor skills. Speech therapy addresses communication difficulties and swallowing disorders, facilitating language and cognitive recovery.  \n  \n Support and Progress  \n  Stroke rehabilitation is a gradual process, and progress varies for each individual. Support from healthcare professionals, caregivers, and support groups is invaluable during this journey. Setting realistic goals and celebrating milestones can boost morale and motivation. With dedication and perseverance, many stroke survivors experience significant improvements in function and quality of life through rehabilitation  \n  \n
            Enter the name of a term from the HeartPedia to know more about it  """,

            "aphasia" : """ :orange[Aphasia]  \n  Understanding Aphasia  \n Aphasia is a complex neurological condition that affects a person's ability to communicate effectively. It often occurs as a result of brain damage, typically from a stroke, brain injury, or neurological disorder. Individuals with aphasia may struggle with speaking, understanding language, reading, and writing, which can significantly impact their daily lives and interpersonal relationships.  \n - Types and Symptoms  \n -  There are different types of aphasia, each with its unique symptoms and severity. Expressive aphasia, for example, impairs a person's ability to speak fluently and form coherent sentences, while receptive aphasia affects comprehension and understanding of spoken or written language. Additionally, individuals may experience difficulty finding the right words (anomia) or struggle with language repetition. These symptoms vary depending on the location and extent of brain damage.  \n - Treatment and Support  \n  While there is no cure for aphasia, various therapies and interventions can help individuals manage and improve their communication skills. Speech and language therapy, cognitive rehabilitation, and assistive communication devices are commonly used to address aphasia's challenges. Moreover, support groups and counseling provide emotional support and practical strategies for coping with the condition. With proper treatment and support, many individuals with aphasia can regain some level of language function and lead fulfilling lives.  \n - Prompt recognition of heart attack symptoms and calling emergency services (911) is crucial for timely treatment.  \n - Treatment options include medications (aspirin, nitroglycerin), angioplasty, or coronary artery bypass surgery.  \n - Rehabilitation and lifestyle changes (diet, exercise, smoking cessation) are vital for recovery and prevention of future events.  \n
            Enter the name of a term from the HeartPedia to know more about it.""",


            "nih stroke scale": """ :orange[NIH Stroke Scale[NISS]]  \n  \n :orange[Understanding NIH Stroke Scale]  \n  The NIH Stroke Scale (NIHSS) is a standardized tool used by healthcare professionals to assess the severity of stroke and guide treatment decisions. It evaluates various neurological functions, including consciousness, sensation, movement, speech, and visual fields. The NIHSS score helps determine the type of stroke, predict outcomes, and monitor response to treatment.  \n  \n  :orange[Scoring and Interpretation]  \n   The NIHSS consists of 11 items, each scored based on the patient's performance during the evaluation. Higher scores indicate greater stroke severity, with a maximum score of 42. Healthcare providers use the NIHSS score to classify strokes as mild, moderate, or severe and tailor treatment accordingly.  \n  \n :orange[Clinical Utility]  \n The NIH Stroke Scale provides valuable information for clinicians assessing stroke patients in various settings, including emergency departments, stroke units, and rehabilitation facilities. It facilitates communication among healthcare team members and ensures consistent evaluation and management practices. Additionally, the NIHSS score serves as a prognostic indicator, helping healthcare providers make informed decisions about patient care and rehabilitation  \n  \n
            Enter the name of a term from the HeartPedia to know more about it  """,


            "nihss": """ :orange[NIH Stroke Scale[NISS]]  \n  \n :orange[Understanding NIH Stroke Scale]  \n  The NIH Stroke Scale (NIHSS) is a standardized tool used by healthcare professionals to assess the severity of stroke and guide treatment decisions. It evaluates various neurological functions, including consciousness, sensation, movement, speech, and visual fields. The NIHSS score helps determine the type of stroke, predict outcomes, and monitor response to treatment.  \n  \n  :orange[Scoring and Interpretation]  \n   The NIHSS consists of 11 items, each scored based on the patient's performance during the evaluation. Higher scores indicate greater stroke severity, with a maximum score of 42. Healthcare providers use the NIHSS score to classify strokes as mild, moderate, or severe and tailor treatment accordingly.  \n  \n :orange[Clinical Utility]  \n The NIH Stroke Scale provides valuable information for clinicians assessing stroke patients in various settings, including emergency departments, stroke units, and rehabilitation facilities. It facilitates communication among healthcare team members and ensures consistent evaluation and management practices. Additionally, the NIHSS score serves as a prognostic indicator, helping healthcare providers make informed decisions about patient care and rehabilitation  \n  \n
            Enter the name of a term from the HeartPedia to know more about it  """,



            "ischemic stroke": """ :orange[About and Facts]  \n Ischemic stroke is the most common type of stroke, accounting for about 87% of all stroke cases. It occurs when a blood vessel supplying blood to the brain is obstructed, usually by a blood clot. This blockage prevents blood and oxygen from reaching the brain's tissues, leading to cell death and potential brain damage. There are two main types of ischemic strokes: thrombotic, caused by a clot forming in an artery directly leading to the brain, and embolic, where a clot forms elsewhere in the body and travels to the brain.  \n  \n  :orange[Symptoms:]  \n - Cardiac arrest is different from a heart attack and requires immediate intervention with CPR and use of an automated external defibrillator (AED).  \n - Survival rates decrease by 7-10% for every minute without CPR or defibrillation.  \n -  Effective bystander CPR can significantly increase the chances of survival and reduce disability.  \n
            Enter the name of a term from the HeartPedia  to know more about it.""",



            


            

            # Add more festivals and their information here
        }

        user_input_lower = user_input.lower()

        if user_input_lower in pedia_info:
            translator = Translator()
            translation = translator.translate(pedia_info[user_input_lower], dest=target_language)
            return translation.text
        else:
            return "I'm sorry, I'm not sure which term you're asking about. Can you please provide more details?"

    if "fest_messages" not in st.session_state:
        st.session_state.fest_messages = [{"role": "assistant", "content": "Discover important stroke-related terms and their meanings with our chatbot!"}]

    for message in st.session_state.fest_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter the Term name"):
        st.session_state.fest_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        language_code = target_languages[target_language]
        response = get_pedia_info(prompt, target_language=language_code)

        with st.spinner(text="Thinking..."):
            st.session_state.fest_messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response, unsafe_allow_html=True)

    

if __name__ == "__main__":
    app()