from gtts import gTTS
from io import BytesIO


def GoogleTTS(language: str = 'ru'):
    """
    google_woman_voice
    """
    def fun(text: str):
        tts = gTTS(text, lang=language)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp

    return fun
