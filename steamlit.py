import streamlit as st
from textblob import TextBlob
from gpt4all import GPT4All

# Initialize the GPT model (update path as necessary)
gpt = GPT4All(model_name="ggml-gpt4all-j-v1.2-jazzy.bin", model_path="D:/model/")

# Function to handle sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment

# Chat handling function
def handle_chat(message):
    if "sentiment" in message.lower():
        # This block will execute in the Streamlit interface when a file is uploaded
        uploaded_file = st.file_uploader("Please upload a text file to analyse", type=['txt'])
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8")
            sentiment = analyze_sentiment(text)
            st.write(f"Analysis of uploaded text:\n{text}")
            st.write(f"Sentiment result: {sentiment}")
    else:
        # Handling general chat using the GPT model
        result = gpt.chat_completion([{"role": "assistant", "content": message}])
        response = result["choices"][0]["message"]["content"]
        return response

def main():
    st.title("MR AI Chatbot")
    user_input = st.text_input("Say something to the AI:")
    if user_input:
        response = handle_chat(user_input)
        if response:
            st.write("AI Response:", response)

if __name__ == "__main__":
    main()
