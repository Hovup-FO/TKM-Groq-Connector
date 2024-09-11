import os
from io import BytesIO
import base64
import requests
import chainlit as cl
from dotenv import load_dotenv
from groq import Groq
from chainlit.input_widget import Select

load_dotenv()

# Groq API keys and settings
groq_api_key = os.getenv("GROQ_API_KEY")
API_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
TEXT_MODEL_ID = "llama-3.1-70b-versatile"  # Default text model ID
VISION_MODEL_ID = "llava-v1.5-7b-4096-preview"
AUDIO_MODEL_ID = "whisper-large-v3"  # Audio model ID
CURRENT_MODEL_ID = TEXT_MODEL_ID # Default to text model on startup

# Initialize Groq client
client = Groq(api_key=groq_api_key)
session_context = {"text": None}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_uploaded_file(file):
    if "image" in file.mime:
        base64_image = encode_image(file.path)
        return "image", base64_image
    elif "audio" in file.mime:
        audio_buffer = BytesIO(file.get_raw_data())
        audio_buffer.seek(0)
        audio_file = audio_buffer.read()
        return "audio", audio_file
    return None, None

@cl.on_chat_start
async def start():
    global CURRENT_MODEL_ID

    elements = [
        cl.Pdf(name="brochure", display="side", path="./docs/brochure.pdf"),
        cl.Video(name="video", url="https://www.youtube.com/watch?v=vyMkqxCOVPY", display="side")
    ]

    await cl.Message(content="I am ready to answer...or take a look at our brochure, or video, if you want to record an audio message, press the recording button (within the prompt) to start recording and press the red recording gadget to stop it. The audio will be sent to us for conversion to text. Or upload an image to ask for its analysis:", elements=elements).send()

    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Groq - Text Models",
                values=[TEXT_MODEL_ID, "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"],
                initial_index=0,
            ),
            Select(
                id="Vision Model",
                label="Groq - Vision Models",
                values=[VISION_MODEL_ID],
                initial_index=0
            ),
            Select(
                id="Audio Model",
                label="Groq - Audio Models",
                values=[AUDIO_MODEL_ID, "distil-whisper-large-v3-en"],
                initial_index=0
            )
        ]
    ).send()
    CURRENT_MODEL_ID = settings["Model"]

async def send_image_to_model(base64_image, user_message):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model=VISION_MODEL_ID,  # Use the vision model for image processing
    )

    response_content = chat_completion.choices[0].message.content
    # Store the response context in the session
    session_context["vision"] = {"role": "assistant", "content": response_content}

    # Inform the user and reset to default text model
    await cl.Message(content="For the moment, our vision model only allows for one message per image.").send()
    global CURRENT_MODEL_ID
    CURRENT_MODEL_ID = TEXT_MODEL_ID

    return response_content

@cl.step(type="tool")
async def speech_to_text(audio_file):
    headers = {
        'Authorization': f'Bearer {groq_api_key}'
    }
    files = {
        'file': ('audio_temp.wav', audio_file),
        'model': (None, AUDIO_MODEL_ID),
        'response_format': (None, 'text'),
        'language': (None, 'en', 'es')
    }

    response = requests.post(API_ENDPOINT, headers=headers, files=files)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print("Details of Error:", response.text)
        raise e

    return response.text

@cl.step(type="tool")
async def generate_text_answer(transcription):
    global session_context
    global CURRENT_MODEL_ID
    messages = [{"role": "user", "content": transcription}]
    response = client.chat.completions.create(
        messages=messages, model=CURRENT_MODEL_ID, temperature=0.3
    )

    response_content = response.choices[0].message.content
    # Store the response context in the session
    session_context["text"] = {"role": "assistant", "content": response_content}
    return response_content

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    audio_buffer = cl.user_session.get("audio_buffer")
    if audio_buffer is None:
        audio_buffer = BytesIO()
        cl.user_session.set("audio_buffer", audio_buffer)
    audio_buffer.write(chunk.data)

@cl.on_audio_end
async def on_audio_end():
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    if audio_buffer is None:
        await cl.Message(content="No audio buffer found.").send()
        return

    audio_buffer.seek(0)
    audio_file = audio_buffer.read()

    # Transcribe the audio
    transcription = await speech_to_text(audio_file)

    # Show the transcription to the user
    await cl.Message(content=f"Transcription: {transcription}").send()

    # Generate a text response
    text_answer = await generate_text_answer(transcription)

    # Send the text response
    await cl.Message(content=text_answer).send()

    # Reset the audio buffer
    cl.user_session.set("audio_buffer", None)

@cl.on_message
async def main(message: cl.Message):
    global CURRENT_MODEL_ID

    if not message.elements:
        # Process text message
        if CURRENT_MODEL_ID is None:
            await cl.Message(content="The model is not selected.").send()
            return

        messages = [{"role": "user", "content": message.content}]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=CURRENT_MODEL_ID,
        )

        response_content = chat_completion.choices[0].message.content
        await cl.Message(content=response_content).send()
    else:
        for element in message.elements:
            file_type, file_content = process_uploaded_file(element)
            if file_type == "image":
                # Use the user's message content for the image analysis
                user_message = message.content.strip()
                if not user_message:
                    user_message = "Can you analyze this image?"  # Fallback message if user doesn't provide one

                chat_completion = await send_image_to_model(file_content, user_message)
                await cl.Message(content=chat_completion).send()
                # Prompt the user to continue with the default text model or re-upload an image
                res = await cl.AskUserMessage(content="Would you like to continue with text or upload another image?", timeout=30, raise_on_timeout=False).send()
                if res:
                    user_response = res['output'].strip().lower()
                    if "upload" in user_response:
                        await cl.Message(content="Please upload a new image.").send()
                    else:
                        CURRENT_MODEL_ID = TEXT_MODEL_ID
                        await cl.Message(content="Switching to text model.").send()
                else:
                    CURRENT_MODEL_ID = TEXT_MODEL_ID
                    await cl.Message(content="No response received. Switching to text model.").send()
            elif file_type == "audio":
                transcription = await speech_to_text(file_content)

                # Show the transcription to the user
                await cl.Message(content=f"Transcription: {transcription}").send()

                text_answer = await generate_text_answer(transcription)
                await cl.Message(content=text_answer).send()

                # Clear transcription context after sending message
                session_context["text"] = None