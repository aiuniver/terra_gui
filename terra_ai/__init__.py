from pathlib import Path
from dotenv import load_dotenv


def settings_load(path: str = ".env"):
    load_dotenv(str(Path(path).absolute()))
