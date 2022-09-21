This is the code for a project I did in my Masters in Product Design: https://louiegavin.com/so-muse/

Basically the idea is to have a physical interface (in this case a social robot) that mainly communicates through voice with a virtual assistant.

It uses the Google Speech API for Speech to Text/Text to Speech and connects with the OpenAI API to use the GPT-3 language model as a chatbot to discuss creative ideas.

The core hardware that was used: Raspberrypi 4B 8GB RAM; ReSpeaker Mic Array v2.0.

The main dependencies might be tricky to install as this seems to we very different on Raspberry Pis. These you need:

pip install openai

pip install google-cloud-speech
pip install --upgrade google-cloud-texttospeech

pip install pygame (for playing the mp3)
