from gtts import gTTS


def GoogleTTS(language: str = 'ru'):
    """
    google_woman_voice
    """
    def fun(text: str):
        tts = gTTS(text, lang=language)
        return tts
    return fun
