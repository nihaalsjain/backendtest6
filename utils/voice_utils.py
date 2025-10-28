import logging
import speech_recognition as sr
import pyttsx3

logger = logging.getLogger(__name__)

def speak(text: str):
    """Converts text to speech and plays it."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        logger.info(f"\n🤖 Assistant:\n{text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"TTS engine error: {e}")

def listen_for_command() -> str:
    """Listens for a command from the microphone and returns it as text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        logger.info("\n🎙️ Listening...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = r.listen(source)
            logger.info("🔍 Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            logger.info(f"🧑 You said: {query}\n")
            return query.lower()
        except sr.UnknownValueError:
            logger.warning("Could not understand audio, please try again.")
            return ""
        except sr.RequestError as e:
            logger.error(f"Speech Recognition request error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return ""
