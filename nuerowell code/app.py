import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import re

st.set_page_config(layout="wide", page_title="NeuroWell", page_icon="🧠")

# Reducing whitespace on the top of the page
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Set up the session state
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

if 'user_type' not in st.session_state:
    st.session_state.user_type = None

@st.cache_data
def is_valid_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

@st.cache_data
def get_slideshow_html():
    return """
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Verdana, sans-serif; }
    .mySlides { display: none; }
    img { vertical-align: middle; width: 100%; margin: 0; padding: 0; }
    .slideshow-container { max-width: 400px; max-height: 400px; position: 100%; margin: 0; }
    .numbertext { color: #f2f2f2; font-size: 12px; padding: 8px 12px; position: absolute; top: 0; }
    .fade { animation-name: fade; animation-duration: 1.9s; }
    @keyframes fade { from {opacity: .4} to {opacity: 1} }
    </style>
    </head>
    <body>
    <div class="slideshow-container">
      <div class="mySlides fade"><div class="numbertext">1 / 4</div><img src="https://www.hiranandanihospital.org/public/uploads/blogs/1691067725.jpg"></div>
      <div class="mySlides fade"><div class="numbertext">2 / 4</div><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvCUh1GDnQgqs_1DZ0PbiconH4UKIjl5EOiQ&s"></div>
      <div class="mySlides fade"><div class="numbertext">3 / 4</div><img src="https://risehealthgroup.com.au/wp-content/uploads/2022/03/nerve-pain.jpg"></div>
      <div class="mySlides fade"><div class="numbertext">4 / 4</div><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR0Y5NNfX-nqjUAsSZS9__gutqQMxrGZDmTRw&s"></div>
    </div>
    <script>
      let slideIndex = 0;
      showSlides();
      function showSlides() {
        let i;
        let slides = document.getElementsByClassName("mySlides");
        for (i = 0; i < slides.length; i++) { slides[i].style.display = "none"; }
        slideIndex++;
        if (slideIndex > slides.length) { slideIndex = 1 }
        slides[slideIndex-1].style.display = "block";
        setTimeout(showSlides, 2000);
      }
    </script>
    </body>
    </html>
    """

if st.session_state.user_info is None:
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
        <div class="gradient-text">Welcome to Neurowell</div>
        """
    st.markdown(gradient_text_html, unsafe_allow_html=True)
    st.write(":orange[Your neuro-rehabilitation platform. Recovering from a stroke, brain injury, or other neurological conditions, we offer personalized plans, virtual therapy, and support tools]")

    st.image('divider.png')

    st.write('Neurological disorders present immense challenges in rehabilitation due to the complexity of physical and mental hurdles, exacerbated by the lack of personalized and integrated solutions. Traditional approaches often fall short in addressing diverse patient needs, with limited accessibility and fragmented care hindering progress. Neurowell aims to revolutionize neuro-rehabilitation by offering personalized care plans, technology-driven accessibility, integration of rehabilitation efforts, and data-driven insights.')
    
    col4, col5, col7, col6, col8, col9, col10 = st.columns([0.03, 0.45, 0.03, 0.6, 0.03, 0.40, 0.02])
    with col5:
        st.write('')
        st.write('')
        components.html(get_slideshow_html(), height=300, width=400)
        typing_animation = """
            <h6 style="text-align: left;">
            <img src="https://readme-typing-svg.herokuapp.com/?font=Righteous&size=35&Left=true&vLeft=true&width=500&height=47&lines=We Care++We Rehab🧠" alt="Typing Animation" />
            </h6>
            """
        st.markdown(typing_animation, unsafe_allow_html=True)

    with col6:
        st.write(" ")
        st.subheader(":orange[Our features]")
        st.write(" - :blue[Comprehensive Stroke Assessment] - Conduct thorough assessments using computer vision games, speech analysis, and memory matching games.")
        st.write(" - :blue[Easy Patient Management] -  Seamlessly create and manage patient profiles with detailed medical histories and previous assessments.")
        st.write(" - :blue[Advanced Diagnostic Insights]-  Receive detailed analysis of stroke type, affected brain areas, and patient performance with graphical representations")
        st.write(" - :blue[Personalized Rehab Plans] - Get tailored rehabilitation plans ranging from independent exercises to motor impulse support and exoskeleton aid.")

    user_role = col9.selectbox('Are you a nurse or a patient?', ('Nurse', 'Patient'))

    do_you_have_an_account = col9.selectbox('Do you have an account?', options=('Yes', 'No', 'I forgot my password'))

    auth_form = col9.form(key='Authentication form', clear_on_submit=False)
    email = auth_form.text_input(label='Email')
    password = auth_form.text_input(label='Password', type='password') if do_you_have_an_account in {'Yes', 'No'} else auth_form.empty()
    auth_notification = col9.empty()

    if do_you_have_an_account == 'Yes' and auth_form.form_submit_button(label='Sign In', use_container_width=True, type='primary'):
        if not email or not password:
            auth_notification.warning("Both fields must be filled.")
        elif not is_valid_email(email):
            auth_notification.warning("Please enter a valid email address.")
        else:
            st.session_state.user_info = email
            st.session_state.user_type = user_role
            st.experimental_rerun()

    elif do_you_have_an_account == 'No' and auth_form.form_submit_button(label='Create Account', use_container_width=True, type='primary'):
        if not email or not password:
            auth_notification.warning("Both fields must be filled.")
        elif not is_valid_email(email):
            auth_notification.warning("Please enter a valid email address.")
        else:
            st.session_state.user_info = email
            st.session_state.user_type = user_role
            st.experimental_rerun()

    elif do_you_have_an_account == 'I forgot my password' and auth_form.form_submit_button(label='Send Password Reset Email', use_container_width=True, type='primary'):
        if not email:
            auth_notification.warning("Email field must be filled.")
        elif not is_valid_email(email):
            auth_notification.warning("Please enter a valid email address.")
        else:
            auth_notification.success("Password reset link has been sent to your email.")

    st.image('divider.png')

    col1, col2, col3 = st.columns([0.6, 1, 0.4])
    with col2:
        st.markdown('##### Project by team - :orange[BYTEBUDDIES] - Deekshith B, Sanjana W G')

else:
    if st.session_state.user_type == 'Nurse':
        import home, physio, hand, game, result

        class MultiApp:
            def __init__(self):
                self.app = []

            def add_app(self, title, func):
                self.app.append({"title": title, "function": func})   

            def run(self):
                with st.sidebar:
                    st.markdown("""
                    <style>
                        .gradient-text { margin-top: -20px; }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    typing_animation = """
                    <h3 style="text-align: left;">
                    <img src="https://readme-typing-svg.herokuapp.com/?font=Righteous&size=35&Left=true&vLeft=true&width=500&height=70&lines=Neuro++Well🧠" alt="Typing Animation" />
                    </h3>
                    """
                    st.markdown(typing_animation, unsafe_allow_html=True)
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    
                    app = option_menu(
                        menu_title='Sections',
                        options=['Home', 'Physio🏋‍♂', 'hand🏋‍♂', 'game🐍', 'Result'],
                        default_index=0,
                        key='nurse_menu' 
                    )
                    
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    
                    linkedin_url = "https://www.linkedin.com/in/deekshith2912/"
                    linkedin_link = f"[ByteBuddies]({linkedin_url})"
                    st.sidebar.header(f"Developed by {linkedin_link}")
                    
                if app == "Home":
                    home.app()
                elif app == "Physio🏋‍♂":
                    physio.app()
                elif app == "hand🏋‍♂":
                    hand.app()
                elif app == "game🐍":
                    game.app()
                elif app == "Result":
                    result.app()

        MultiApp().run()

    elif st.session_state.user_type == 'Patient':
        import homee, rehab, community

        class MultiApp:
            def __init__(self):
                self.app = []

            def add_app(self, title, func):
                self.app.append({"title": title, "function": func})

            def run(self):
                with st.sidebar:
                    st.markdown("""
                    <style>
                        .gradient-text { margin-top: -20px; }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    typing_animation = """
                    <h3 style="text-align: left;">
                    <img src="https://readme-typing-svg.herokuapp.com/?font=Righteous&size=35&Left=true&vLeft=true&width=500&height=70&lines=Neuro++Well🧠" alt="Typing Animation" />
                    </h3>
                    """
                    st.markdown(typing_animation, unsafe_allow_html=True)
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    
                    app = option_menu(
                        menu_title='Sections',
                        options=['Homee', 'Rehab🏋‍♂', 'Community'],
                        default_index=0,
                        key='patients_menu'                          #give different values when you get same button error
                    )
                    
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    
                    linkedin_url = "https://www.linkedin.com/in/deekshith2912/"
                    linkedin_link = f"[ByteBuddies]({linkedin_url})"
                    st.sidebar.header(f"Developed by {linkedin_link}")
                    
                if app == "Homee":
                    homee.app()
                elif app == "Rehab🏋‍♂":
                    rehab.app()
                elif app == "Community":
                    community.app()
                
        MultiApp().run()
