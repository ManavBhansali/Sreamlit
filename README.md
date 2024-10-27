Streamlit Multilingual Translator and Speech Processor

A user-friendly, multilingual Streamlit app designed to seamlessly integrate Text-to-Speech, Speech-to-Text, Text Translation, and Multi-Speaker Translation functionalities. This project enables cross-lingual communication and provides robust translation and speech tools, all in a single interface.

<!-- Optional: Add a screenshot of your app here -->
Features

    Text-to-Speech: Type text in any supported language and have it read aloud.
    Speech-to-Text: Capture spoken input from your microphone and convert it to text in real-time.
    Text-to-Text Translation: Translate text from one language to another without needing audio.
    Multi-Speaker Functionality: Enable two-way conversations between users who speak different languages. Each speaker can select a language, and the system will handle translation back and forth, allowing seamless cross-lingual communication.

Table of Contents

    Installation
    Usage
    Technologies
    Project Structure
    Acknowledgments
    License

Installation

    Clone the Repository:

    bash

git clone https://github.com/username/repo-name.git
cd repo-name

Install Required Packages: Ensure that you have Python installed, then run:

bash

pip install -r requirements.txt

Run the Streamlit App:

bash

    streamlit run app.py

Usage

    Text-to-Speech:
        Enter text into the provided text box.
        Choose the language and click Convert to Speech.

    Speech-to-Text:
        Click Start Recording and speak into your microphone.
        The app will display the recognized text.

    Text-to-Text Translation:
        Enter the text in the source language, select the target language, and click Translate.

    Multi-Speaker Translation:
        Each speaker selects their source and target languages.
        Speak into your respective microphone, and the app will display and translate each speaker’s input for seamless communication.

Technologies

    Streamlit: For creating the web application.
    SpeechRecognition: For capturing and converting speech to text.
    pyttsx3: For text-to-speech capabilities.
    Transformers (Hugging Face): For natural language processing and translation.
    Other Libraries: Standard Python libraries for text handling, I/O, and data management.

Project Structure

plaintext

repo-name/
├── app.py                  # Main Streamlit application
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
└── assets/
    └── screenshot.png      # Optional screenshots or assets

Acknowledgments

This project was inspired by the need for easy cross-lingual communication tools and was built using open-source technologies from the Python community.
License

This project is licensed under the MIT License - see the LICENSE file for details.
