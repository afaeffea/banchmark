"""Клиент Yandex SpeechKit (REST API v1: синхронные stt:recognize / tts:synthesize).

Документация:
- STT: https://yandex.cloud/en/docs/speechkit/stt/request (max 1MB / 30s, mono, LPCM или OggOpus)
- TTS: https://aistudio.yandex.ru/docs/en/speechkit/quickstart/tts-quickstart-v1.html
"""
from __future__ import annotations

import time

import requests

from common.env import get_env, require_env
from .base import STTProvider, STTResult, TTSProvider, TTSResult

_STT_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
_TTS_URL = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"


class YandexSTT(STTProvider):
    name = "yandex"

    def __init__(self, lang: str = "ru-RU"):
        self.api_key = require_env("YANDEX_API_KEY")
        self.folder_id = require_env("YANDEX_FOLDER_ID")
        self.lang = lang

    def recognize(self, audio_bytes: bytes, sample_rate_hz: int = 16000) -> STTResult:
        params = {
            "folderId": self.folder_id,
            "lang": self.lang,
            "format": "lpcm",
            "sampleRateHertz": str(sample_rate_hz),
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        t0 = time.perf_counter()
        try:
            resp = requests.post(_STT_URL, params=params, headers=headers, data=audio_bytes, timeout=30)
            latency = time.perf_counter() - t0
            if resp.status_code != 200:
                return STTResult(text="", latency_s=latency, raw_error=f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            return STTResult(text=data.get("result", ""), latency_s=latency)
        except requests.RequestException as exc:
            return STTResult(text="", latency_s=time.perf_counter() - t0, raw_error=str(exc))


class YandexTTS(TTSProvider):
    name = "yandex"

    def __init__(self, voice: str = "filipp", lang: str = "ru-RU", sample_rate_hz: int = 16000):
        self.api_key = require_env("YANDEX_API_KEY")
        self.folder_id = require_env("YANDEX_FOLDER_ID")
        self.voice = get_env("YANDEX_VOICE", voice)
        self.lang = lang
        self.sample_rate_hz = sample_rate_hz

    def synthesize(self, text: str) -> TTSResult:
        data = {
            "text": text,
            "lang": self.lang,
            "voice": self.voice,
            "folderId": self.folder_id,
            "format": "lpcm",
            "sampleRateHertz": str(self.sample_rate_hz),
        }
        headers = {"Authorization": f"Api-Key {self.api_key}"}
        t0 = time.perf_counter()
        try:
            resp = requests.post(_TTS_URL, headers=headers, data=data, timeout=30)
            latency = time.perf_counter() - t0
            if resp.status_code != 200:
                return TTSResult(audio_bytes=b"", audio_format="pcm16", latency_s=latency,
                                  raw_error=f"HTTP {resp.status_code}: {resp.text[:500]}")
            return TTSResult(audio_bytes=resp.content, audio_format="pcm16", latency_s=latency)
        except requests.RequestException as exc:
            return TTSResult(audio_bytes=b"", audio_format="pcm16", latency_s=time.perf_counter() - t0,
                              raw_error=str(exc))
