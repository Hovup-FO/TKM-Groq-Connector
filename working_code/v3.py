import os
from io import BytesIO
import base64
import requests
import chainlit as cl
from dotenv import load_dotenv
from groq import Groq
from chainlit.input_widget import Select
from PIL import Image
import pyheif

load_dotenv()

# Groq API keys y configuración
groq_api_key = os.getenv("GROQ_API_KEY")
API_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
TEXT_MODEL_ID = "llama-3.1-70b-versatile"  # Default text model ID
VISION_MODEL_ID = "llava-v1.5-7b-4096-preview"
AUDIO_MODEL_ID = "whisper-large-v3"  # Audio model ID
CURRENT_MODEL_ID = TEXT_MODEL_ID  # Default to text model on startup

# Inicializar el cliente de Groq
client = Groq(api_key=groq_api_key)
session_context = {"text": None}

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            print("Image encoding to base64 successful")
            return base64_image
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def convert_heic_to_jpeg(heic_file_path):
    try:
        print(f"Converting HEIC file: {heic_file_path}")
        heif_file = pyheif.read(heic_file_path)
        image = Image.frombytes(
            mode=heif_file.mode,
            size=heif_file.size,
            data=heif_file.data,
            decoder_name="raw"
        )

        jpeg_bytes = BytesIO()
        image.save(jpeg_bytes, format="JPEG")
        jpeg_bytes.seek(0)
        print("HEIC file conversion to JPEG successful")
        base64_image = base64.b64encode(jpeg_bytes.read()).decode('utf-8')
        print("JPEG file encoding to base64 successful")
        return base64_image
    except Exception as e:
        print(f"Error converting HEIC to JPEG: {e}")
        return None

def convert_png_to_jpeg(png_file_path):
    try:
        print(f"Converting PNG file: {png_file_path}")
        image = Image.open(png_file_path)
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        jpeg_bytes = BytesIO()
        image.save(jpeg_bytes, format="JPEG")
        jpeg_bytes.seek(0)
        print("PNG file conversion to JPEG successful")
        base64_image = base64.b64encode(jpeg_bytes.read()).decode('utf-8')
        print("JPEG file encoding to base64 successful")
        return base64_image
    except Exception as e:
        print(f"Error converting PNG to JPEG: {e}")
        return None

async def process_uploaded_file(file):
    async with cl.Step(name="File Reception", type="tool") as step:
        step.input = f"File received: {file.name} with mime type {file.mime}"
        print(step.input)
        if "image" in file.mime or file.mime == "application/octet-stream":
            if file.mime == "image/heic" or file.name.lower().endswith(".heic"):
                async with cl.Step(name="Converting HEIC to JPEG", type="tool") as convert_step:
                    convert_step.input = "Processing HEIC file..."
                    print(convert_step.input)
                    base64_image = convert_heic_to_jpeg(file.path)
                    if base64_image is None:
                        raise ValueError("Conversion returned None")
                    convert_step.output = "HEIC converted and base64 encoded successfully"
                    print(convert_step.output)
                    return "image", base64_image
            elif file.mime == "image/png" or file.name.lower().endswith(".png"):
                async with cl.Step(name="Converting PNG to JPEG", type="tool") as convert_step:
                    convert_step.input = "Processing PNG file..."
                    print(convert_step.input)
                    base64_image = convert_png_to_jpeg(file.path)
                    if base64_image is None:
                        raise ValueError("Conversion returned None")
                    convert_step.output = "PNG converted and base64 encoded successfully"
                    print(convert_step.output)
                    return "image", base64_image
            else:
                async with cl.Step(name="Encoding Image to Base64", type="tool") as encode_step:
                    encode_step.input = f"Processing {file.mime} file..."
                    print(encode_step.input)
                    base64_image = encode_image(file.path)
                    if base64_image is None:
                        raise ValueError("Encoding returned None")
                    encode_step.output = f"{file.mime} converted and base64 encoded successfully"
                    print(encode_step.output)
                    return "image", base64_image
        elif "audio" in file.mime:
            async with cl.Step(name="Processing Audio File", type="tool") as audio_step:
                audio_step.input = "Processing audio file..."
                print(audio_step.input)
                audio_buffer = BytesIO(file.get_raw_data())
                audio_buffer.seek(0)
                audio_file = audio_buffer.read()
                audio_step.output = "Audio file processed successfully"
                print(audio_step.output)
                return "audio", audio_file
        step.output = "File type not supported"
        print(step.output)
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
    async with cl.Step(name="Send Image to Model", type="llm") as step:
        step.input = "Sending image to vision model..."
        print(step.input)
        try:
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
            await cl.Message(content="For the moment our vision model only allows for one analysis message per image.").send()

            global CURRENT_MODEL_ID
            CURRENT_MODEL_ID = TEXT_MODEL_ID

            step.output = response_content
            print(step.output)
            return response_content
        except Exception as e:
            error_message = f"Error sending image to model: {e}"
            print(error_message)
            step.output = error_message
            return None

@cl.step(type="tool")
async def speech_to_text(audio_file):
    async with cl.Step(name="Speech to Text", type="tool") as step:
        step.input = "Processing audio to text..."
        print(step.input)
        try:
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
            response.raise_for_status()  # Raise an error for bad status codes
            step.output = response.text
            print(step.output)
            return response.text
        except requests.exceptions.HTTPError as e:
            error_message = f"HTTP error occurred: {e}"
            print(error_message)
            step.output = error_message
            return None
        except Exception as e:
            error_message = f"Error processing audio to text: {e}"
            print(error_message)
            step.output = error_message
            return None

@cl.step(type="tool")
async def generate_text_answer(transcription):
    async with cl.Step(name="Generate Text Answer", type="tool") as step:
        step.input = transcription
        print(step.input)
        try:
            global session_context
            global CURRENT_MODEL_ID
            messages = [{"role": "user", "content": transcription}]
            response = client.chat.completions.create(
                messages=messages, model=CURRENT_MODEL_ID, temperature=0.3
            )

            response_content = response.choices[0].message.content
            # Store the response context in the session
            session_context["text"] = {"role": "assistant", "content": response_content}
            step.output = response_content
            print(step.output)
            return response_content
        except Exception as e:
            error_message = f"Error generating text answer: {e}"
            print(error_message)
            step.output = error_message
            return None

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

    transcription = await speech_to_text(audio_file)

    if transcription:
        await cl.Message(content=f"Transcription: {transcription}").send()
        text_answer = await generate_text_answer(transcription)
        await cl.Message(content=text_answer).send()
        cl.user_session.set("audio_buffer", None)
    else:
        await cl.Message(content="Error in audio transcription.").send()

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
            print(f"Processing element of type: {element.mime}")
            file_type, file_content = await process_uploaded_file(element)
            if file_type == "image":
                print("Image processed successfully.")
                if file_content is None:
                    await cl.Message(content=f"Error processing image of type {element.mime}.").send()
                    continue
                user_message = message.content.strip()
                if not user_message:
                    user_message = "Can you analyze this image?"  # Fallback message if user doesn't provide one

                chat_completion = await send_image_to_model(file_content, user_message)
                if chat_completion:
                    await cl.Message(content=chat_completion).send()
                else:
                    await cl.Message(content="Error analyzing the image.").send()

                res = await cl.AskUserMessage(content="Would you like to continue with vision analysis or switch to text based conversations?", timeout=60, raise_on_timeout=False).send()
                if res:
                    user_response = res['output'].strip().lower()
                    if "vision" in user_response:
                        await cl.Message(content="Please upload a new image.").send()
                    else:
                        CURRENT_MODEL_ID = TEXT_MODEL_ID
                        await cl.Message(content="Switching to text model.").send()
                else:
                    CURRENT_MODEL_ID = TEXT_MODEL_ID
                    await cl.Message(content="No response received. Switching to text model.").send()
            elif file_type == "audio":
                transcription = await speech_to_text(file_content)

                if transcription:
                    await cl.Message(content=f"Transcription: {transcription}").send()
                    text_answer = await generate_text_answer(transcription)
                    await cl.Message(content=text_answer).send()
                    session_context["text"] = None
                else:
                    await cl.Message(content="Error in audio transcription.").send()