## Dhwani - Voice Mode For Kannada

- Dhwani is a self-hosted platform designed to provide Voice mode interaction for Kannada and Indian languages.


- Research Goal -

  - The primary research goal of Project Dhwani is to measure and improve the Time to First Token Generation (TTFTG) for model architectures in Automatic Speech Recognition (ASR), Translation, and Text-to-Speech (TTS) systems. 
  - By leveraging open-source Large Language Models (LLMs) and tools provided by AI4BHARAT, we aim to develop and enhance a Kannada voice model that meets industry standards set by Alexa, Siri, and Google. 
  - The project will focus on creating robust voice solutions for Indian languages, with a specific emphasis on Kannada.


- Project Report 
  - [WebLink](https://github.com/sachinsshetty/onwards/blob/main/idea/2025/2025-02-24-gpu-access.md) 
  - [Document](https://docs.google.com/document/d/1idZAzXc65e5QtwTO4vW8gImqnKets_4N4OHLZOCZ9Q0/edit?tab=t.0) 


  ### Models and Tools
The project will utilize the following open-source tools:

| Open-Source Tool                       | Source Repository                                          | CPU / Available 24/7 - Free, Slow | GPU / Paused, On-demand, $.05 /hour |
|---------------------------------------|-------------------------------------------------------------|----------------|----------------|
| Automatic Speech Recognition : ASR   | [ASR Indic Server](https://github.com/slabstech/asr-indic-server) | [HF Demo](https://huggingface.co/spaces/gaganyatri/asr_indic_server_cpu) |  [Ondemand - HF Demo](https://huggingface.co/spaces/gaganyatri/asr_indic_server)  |
| Text to Speech : TTS                  | [TTS Indic Server](https://github.com/slabstech/tts-indic-server)  | Not suitable           | [Ondemand - HF Demo](https://huggingface.co/spaces/gaganyatri/asr_indic_server) |
| Translation                           | [Indic Translate Server](https://github.com/slabstech/indic-translate-server) | [HF Demo](https://huggingface.co/spaces/gaganyatri/translate_indic_server_cpu)          | [Ondemand - HF Demo](https://huggingface.co/spaces/gaganyatri/translate_indic_server)            |


## Target Solution

| Answer Engine                                  | Voice Translation                          |
|-----------------------------------------------|---------------------------------------------|
| ![Answer Engine](docs/workflow/kannada-answer-engine.drawio.png "Engine") | ![Voice Translation](docs/workflow/voice-translation.drawio.png "Voice Translation") |


- Demo for Testing components for Dhwani for Accuracy and evaluation


| Feature                      | Description                                                                 | Demo Link | Components          | Source Code       |
|------------------------------|-----------------------------------------------------------------------------|-----------|---------------------|-------------------|
| Answer Engine                | Provides answers to queries using a large language model.                     |   | LLM                 | [Link](ux/answer_engine/app.py)          |
| Answer Engine with Translate| Provides answers to queries with translation capabilities.                   | [Link](https://huggingface.co/spaces/gaganyatri/dhwani-voice-model)  | LLM, Translation    | [Link](ux/answer_engine_translate/app.py)          |
| PDF Translate                | Translates content from PDF documents.                                       |  | Translation         |           |
| Text Translate               | Translates text from one language to another.                                |   | Translation         | [Link]()          |
| Voice Generation            | Generates speech from text.                                                  |   | TTS                 | [Link](ux/text_to_speech/app.py)          |
| Voice to Text Translation    | Converts spoken language to text and translates it.                          | [Link](https://huggingface.co/spaces/gaganyatri/dhwani-tts)  | ASR, Translation    | [Link](ux/voice_to_text_translation/app.py)          |
| Voice to Voice Translation   | Converts spoken language to text, translates it, and then generates speech.   | [Link](https://huggingface.co/spaces/gaganyatri/dhwani-tts)  | ASR, Translation, TTS| [Link](ux/voice_to_voice_translation/app.py)          |
| Text Query                   | Allows querying text data for specific information.                          | [Link](https://huggingface.co/spaces/gaganyatri/dhwani_text_query)  | LLM                 | [Link](ux/text_query/app.py)          |


- For collaborations
  - Join the discord group - [invite link](https://discord.gg/WZMCerEZ2P)

- For business queries, Email : info (at) slabstech (dot) com
