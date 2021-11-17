from gtts import gTTS


def google_woman_voice(language: str = 'ru'):
    def fun(text: str):
        tts = gTTS(text, lang=language)
        return tts
    return fun
