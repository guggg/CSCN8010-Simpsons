from PIL import Image
import streamlit as st
from tf_model import model_run

#from dotenv import load_dotenv
#load_dotenv()

###Initial UI configuration:###
st.set_page_config(page_title='Conestoga AIML', page_icon="🦙", layout="wide")


def render_app():
    # reduce font sizes for input text boxes
    custom_css = """
        <style>
            .stTextArea textarea {font-size: 13px;}
            div[data-baseweb="select"] > div {font-size: 13px !important;}
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    #Left sidebar menu
    st.sidebar.header('Conestoga AIML')

    #Set config for a cleaner menu, footer & background:
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    if 'chat_dialogue' not in st.session_state:
        st.session_state['chat_dialogue'] = []

    if 'pre_prompt' not in st.session_state:
        st.session_state['pre_prompt'] = ''

    #Dropdown menu to select the model edpoint:
    selected_option = st.sidebar.selectbox('Choose a Model:', ['CNN', 'MobileNetV2', 'ResNet50'], key='model')
    if selected_option == 'CNN':
        st.session_state['models'] = 'CNN'
    elif selected_option == 'MobileNetV2':
        st.session_state['models'] = 'MobileNetV2'
    elif selected_option == 'ResNet50':
        st.session_state['models'] = 'ResNet50'

    #Model hyper parameters:
    #st.session_state['temperature'] = st.sidebar.slider('Temperature:', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    #st.session_state['top_p'] = st.sidebar.slider('Top P:', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    #st.session_state['max_seq_len'] = st.sidebar.slider('Max Sequence Length:', min_value=64, max_value=4096, value=2048, step=8)

    btn_col1, btn_col2 = st.sidebar.columns(2)

    def clear_history():
        st.session_state['chat_dialogue'] = []
    clear_chat_history_button = btn_col1.button("Clear History",
                                            use_container_width=True,
                                            on_click=clear_history)


    # add links to relevant resources for users to select
    st.sidebar.write(" ")

    text1 = 'llama2-chatbot'
    text2 = 'bckgmail'
    text3 = 'CNN + ResNet'
    text4 = 'The Simpsons Characters Data'

    text1_link = 'https://github.com/a16z-infra/llama2-chatbot'
    text2_link = 'https://github.com/KajPe/bckgmail'
    text3_link = 'https://www.kaggle.com/code/vellyy/cnn-resnet'
    text4_link = 'https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset'

    logo1 = 'https://cdn.pixabay.com/photo/2022/01/30/13/33/github-6980894_1280.png'
    logo2 = 'https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg'
    logo3 = 'https://static-00.iconduck.com/assets.00/kaggle-icon-2048x2048-fxhlmjy3.png'

    st.sidebar.markdown(
        "**Reference Code and Model:**  \n"
        f"<img src='{logo3}' style='height: 2em'> [{text4}]({text4_link})  \n"
        f"<img src='{logo1}' style='height: 2em'> [{text1}]({text1_link})  \n"
        f"<img src='{logo3}' style='height: 2em'> [{text3}]({text3_link})",
        unsafe_allow_html=True)

    icon_arcadio = 'https://avatars.githubusercontent.com/u/122412860?v=4'
    icon_givors = 'https://avatars.githubusercontent.com/u/17698876?v=4'
    icon_kyle = 'https://avatars.githubusercontent.com/u/777378?v=4'

    st.sidebar.write(" ")
    st.sidebar.markdown(
        "**Contributors:**  \n"
#        f"<img src='{icon_arcadio}' style='height: 2em'> [{'**Arcadio**'}]({'https://github.com/arcadiopfz'})  \n"
        f"<img src='{icon_givors}' style='height: 2em'> [{'**Givors Ku**'}]({'https://github.com/guggg'})  \n"
        f"<img src='{icon_kyle}' style='height: 2em'> [{'**Kyle**'}]({'https://github.com/onlyxool'})",
        unsafe_allow_html=True)

    st.title(st.session_state['models'])

    uploaded_file = st.file_uploader("Upload a Image", type=('jpg', 'jpeg', 'png'))

    for message in st.session_state.chat_dialogue:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if uploaded_file:
        if st.session_state['models'] == 'MobileNetV2':
            if uploaded_file.name.lower().endswith('.png'):
                image = Image.open(uploaded_file).convert('RGB').resize((224, 224))
            else:
                image = Image.open(uploaded_file).resize((224, 224))
        else:
            if uploaded_file.name.lower().endswith('.png'):
                image = Image.open(uploaded_file).convert('RGB').resize((180, 180))
            else:
                image = Image.open(uploaded_file).resize((180, 180))
        st.image(image, width=150, channels="BGR")



        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                output = model_run(st.session_state['models'], image)
            message_placeholder.markdown(output)


render_app()