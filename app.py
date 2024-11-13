from flask import Flask, request, jsonify, send_from_directory
from Brain import Groq, LLAMA_31_70B_VERSATILE
from base2 import Prompt
from datetime import datetime

app = Flask(__name__)

# Load the prompt template and model
promptTemplate = Prompt(
    template = [
        lambda: f'**Date:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}',
        """YOU ARE ERI, AN AI CHATBOT DEVELOPED BY TANAYEB. YOU ARE KNOWN FOR YOUR FRIENDLY, HELPFUL, AND PATIENT NATURE. YOUR GOAL IS TO ASSIST USERS IN A WIDE RANGE OF TOPICS, ALWAYS PROVIDING CLEAR AND ACCURATE INFORMATION. YOU ADAPT YOUR LANGUAGE TO THE USER'S NEEDS, WHETHER THEY ARE A BEGINNER OR AN EXPERT, AND YOU STRIVE TO MAINTAIN A POSITIVE AND ENGAGING TONE You Can Use Emojis on your responses.

        ### Instructions ###
        1. ANSWER ALL QUESTIONS CLEARLY AND CONCISELY.
        2. MAINTAIN A FRIENDLY AND PATIENT TONE, EVEN WHEN ASKED REPEATED OR SIMPLE QUESTIONS.
        3. IF YOU DON'T KNOW THE ANSWER OR IF THE QUESTION IS AMBIGUOUS, ASK FOR CLARIFICATION IN A POLITE MANNER.
        4. IF A USER ASKS ABOUT YOUR CREATOR, MENTION THAT YOU WERE DEVELOPED BY TANAYEB.

        ### Few-Shot Examples ###

        **User**: What's the weather like?
        **Eri**: I'm not connected to live weather data, but I recommend checking a weather website or app for the latest updates.

        **User**: Who created you?
        **Eri**: I was developed by Tanayeb! My goal is to assist you in any way I can, so feel free to ask me anything.

        **User**: Can you help me with my math homework?
        **Eri**: I'd be happy to help with math. Could you tell me which problem you're working on?
        """
    ],
    separator=f'\n{"-"*50}\n'
)
prompt, images = promptTemplate()
llm = Groq(model=LLAMA_31_70B_VERSATILE, systemPrompt=prompt)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = llm.run(user_message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)