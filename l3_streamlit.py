import streamlit as st

# Create frontend with Markdown
st.write("""
# Pet Name Generator
**Select the type of pet** you have to get some naming suggestions!
""")

# Or add inputs with custom functions
animal_type = st.selectbox('Pet type', ['cat', 'dog', 'parrot', 'fish'])
animal_type_unvalidated = st.text_input('Pet type', max_chars=50)