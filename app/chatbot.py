import streamlit as st
from utils import *
import time
from gen_utils import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-3B")

# Function to classify text
def classify_text(text):
    # Replace this with your actual classification logic/model
    # For simplicity, let's assume any text mentioning "covid" is classified as a disease
    if "covid" in text.lower():
        return True
    else:
        return False

# context = "You are an AI psychotherapist dedicated to providing emotional support and guidance to users. Your goal is to assist individuals in managing stress, understanding their emotions, and offering coping strategies for various life situations. Users can engage in conversations with you to discuss their feelings, thoughts, and concerns, and you will respond with empathy, understanding, and therapeutic insights. While you can offer support, it's important to remind users that your responses are not a substitute for professional mental health advice, and you may encourage them to seek help from qualified professionals when needed."
# Streamlit UI
def main():
    st.title("Twitter-like UI with Disease Classifier")

    # Text input
    user_input = st.text_area("Enter your tweet:")

    # Check if the input text contains a disease
    is_disease = classify_text(user_input)

    # Display result
    if is_disease:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.success("Disease detected! Opening chatbot...")

        # Chatbot functionality
        st.sidebar.title("Chatbot")
        # if prompt := st.chat_input("How can I help you?"):
        #     st.session_state.messages.append({"role": "user", "content": prompt})

        if prompt := st.chat_input("How can I help you?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Add user message to chat history
            # st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user", avatar="👩‍💻"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                inputs = tokenizer(prompt, return_tensors='pt')
                res = model.generate(**inputs)
                message_placeholder = st.empty()
                full_response = ""
                assistant_response = tokenizer.decode(res[0])
                # Simulate stream of response with milliseconds delay
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)


            # Display assistant response in chat message container
            # with st.chat_message("assistant"):
            #     message_placeholder = st.empty()
            #     full_response = ""
            #     assistant_response = gen_ai(context=None, prompt=prompt)
            #     # Simulate stream of response with milliseconds delay
            #     for chunk in assistant_response.split():
            #         full_response += chunk + " "
            #         time.sleep(0.05)
            #         # Add a blinking cursor to simulate typing
            #         message_placeholder.markdown(full_response + "▌")
            #     message_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

#         # Display chat history
#         # chat_history = st.sidebar.text_area("Chat History", value="Chatbot: There is something off with your tweet. Can I help you in any way?")
#         #
#         # # Receive user input
#         # user_message = st.sidebar.text_input("You:", "")
#         #
#         # # Process user input and update chat history
#         # if user_message:
#         #     chat_history += f"\nYou: {user_message}"
#         #
#         # st.sidebar.text_area("Chat History", value=chat_history)
#
    else:
        st.info("No disease detected.")


if __name__ == "__main__":
    main()
# #
# #
# # # Reference: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
# # # https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/
# # # https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/
# #
# # import streamlit as st
# # import random
# # import time
# # from utils import gen_ai, side_bar
# # import os
# # path =os.getcwd() + os.sep + 'gwu.jpg'
# # left_co, cent_co,last_co = st.columns(3)
# # with cent_co:
# #     logo = st.image(path, width=100)
# #
# # st.title("NLP Class Chatbot with AI")
# #
# # # Initialize chat history
# # if "messages" not in st.session_state:
# #     st.session_state.messages = []
# #
# # # Display chat messages from history on app rerun
# # for message in st.session_state.messages:
# #     with st.chat_message(message["role"], avatar=message.get("avatar", "👤")):
# #         st.markdown(message["content"])
# #
# # side_bar()
# #
# #
# # # Accept user input
# # if prompt := st.chat_input("How can I help you?"):
# #     st.session_state.messages.append({"role": "user", "content": prompt})
# #     # Add user message to chat history
# #     # st.session_state.messages.append({"role": "user", "content": prompt})
# #     # Display user message in chat message container
# #     with st.chat_message("user", avatar="👩‍💻"):
# #         st.markdown(prompt)
# #
# #     # Display assistant response in chat message container
# #     with st.chat_message("assistant"):
# #         message_placeholder = st.empty()
# #         full_response = ""
# #         assistant_response = gen_ai(context=None, prompt=prompt)
# #         # Simulate stream of response with milliseconds delay
# #         for chunk in assistant_response.split():
# #             full_response += chunk + " "
# #             time.sleep(0.05)
# #             # Add a blinking cursor to simulate typing
# #             message_placeholder.markdown(full_response + "▌")
# #         message_placeholder.markdown(full_response)
# #
# #
# #
# #
# #     # Add assistant response to chat history
# #     st.session_state.messages.append({"role": "assistant", "content": full_response})
# #
#
