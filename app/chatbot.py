import streamlit as st


# Function to classify text
def classify_text(text):
    # Replace this with your actual classification logic/model
    # For simplicity, let's assume any text mentioning "covid" is classified as a disease
    if "covid" in text.lower():
        return True
    else:
        return False


# Streamlit UI
def main():
    st.title("Twitter-like UI with Disease Classifier")

    # Text input
    user_input = st.text_area("Enter your tweet:")

    # Check if the input text contains a disease
    is_disease = classify_text(user_input)

    # Display result
    if is_disease:
        st.success("Disease detected! Opening chatbot...")

        # Chatbot functionality
        st.sidebar.title("Chatbot")

        # Display chat history
        chat_history = st.sidebar.text_area("Chat History", value="Chatbot: Hi! I'm the chatbot. How can I help you?")

        # Receive user input
        user_message = st.sidebar.text_input("You:", "")

        # Process user input and update chat history
        if user_message:
            chat_history += f"\nYou: {user_message}"

        st.sidebar.text_area("Chat History", value=chat_history)

    else:
        st.info("No disease detected.")


if __name__ == "__main__":
    main()
