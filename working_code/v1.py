import os
from io import BytesIO
import requests
import chainlit as cl
from dotenv import load_dotenv
from groq import Groq
from chainlit.input_widget import Select

load_dotenv()

# Groq API keys y configuración
groq_api_key = os.getenv("GROQ_API_KEY")
API_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL_ID = "whisper-large-v3"

# Inicializar el cliente de Groq
client = Groq(api_key=groq_api_key)
value = None

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    audio_buffer = cl.user_session.get("audio_buffer")
    if audio_buffer is None:
        audio_buffer = BytesIO()
        cl.user_session.set("audio_buffer", audio_buffer)
    audio_buffer.write(chunk.data)

@cl.step(type="tool")
async def speech_to_text(audio_file):
    headers = {
        'Authorization': f'Bearer {groq_api_key}'
    }
    files = {
        'file': ('audio_temp.wav', audio_file),
        'model': (None, MODEL_ID),
        'response_format': (None, 'text'),
        'language': (None, 'en', 'es')
    }

    response = requests.post(API_ENDPOINT, headers=headers, files=files)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Imprimir detalles del error
        print("Details of Error:", response.text)
        raise e

    return response.text

@cl.step(type="tool")
async def generate_text_answer(transcription):
    messages = [{"role": "user", "content": transcription}]
    response = client.chat.completions.create(
        messages=messages, model=value, temperature=0.3
    )
    return response.choices[0].message.content

@cl.on_audio_end
async def on_audio_end():
    # Obtener el buffer de audio de la sesión
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    if audio_buffer is None:
        await cl.Message(content="No audio buffer found.").send()
        return

    audio_buffer.seek(0)  # Mover el puntero al inicio
    audio_file = audio_buffer.read()

    # Transcribir el audio
    transcription = await speech_to_text(audio_file)

    # Generar respuesta del LLM
    text_answer = await generate_text_answer(transcription)

    # Enviar la respuesta al chat
    await cl.Message(content=text_answer).send()

@cl.on_chat_start
async def start():
    global value
    elements = [
        cl.Pdf(name="brochure", display="side", path="./docs/brochure.pdf"),
        cl.Video(name="video", url="https://www.youtube.com/watch?v=vyMkqxCOVPY", display="side")
    ]
    await cl.Message(content="I am ready to answer...or take a look at our brochure, or video, if you want to record and audio once you press the recording button (wihthin the prompt) it will start recording and until you press the red recording gadget that pops up, it will stop recording and send it to us and we will convert it to text.", elements=elements).send()

    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Groq - Models",
                values=["llama-3.1-70b-versatile", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"],
                initial_index=0,
            )
        ]
    ).send()
    value = settings["Model"]

@cl.on_message
async def main(message: cl.Message):
    global value
    if value is None:
        await cl.Message(content="The model is not selected.").send()
        return

    if message.content.strip():
        # Procesar el mensaje de texto
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message.content,
                }
            ],
            model=value,
        )

        await cl.Message(
            content=f"{chat_completion.choices[0].message.content}",
        ).send()
    else:
        if not message.elements:
            await cl.Message(content="No file attached").send()
            return

        # Procesar archivos de audio adjuntos
        audio_files = [file for file in message.elements if "audio" in file.mime]
        for audio in audio_files:
            audio_buffer = BytesIO(audio.get_raw_data())
            audio_buffer.seek(0)
            transcription = await speech_to_text(audio_buffer.read())
            text_answer = await generate_text_answer(transcription)
            await cl.Message(content=text_answer).send()