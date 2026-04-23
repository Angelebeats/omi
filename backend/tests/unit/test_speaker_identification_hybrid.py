import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import os
import sys

# Mocking database and other modules that might be imported
os.environ.setdefault("ENCRYPTION_SECRET", "omi_ZwB2ZNqB2HHpMK6wStk7sTpavJiPTFg7gXUHnc4tFABPU6pZ2c2DKgehtfgi4RZv")
sys.modules.setdefault("database._client", MagicMock())
sys.modules.setdefault("utils.other.storage", MagicMock())

# Path to the omi project backend
PROJECT_BACKEND = r"C:\Users\Administrator\.accio\accounts\7083453322\agents\DID-F456DA-2B0D4C\project\omi\backend"
if PROJECT_BACKEND not in sys.path:
    sys.path.insert(0, PROJECT_BACKEND)

from utils.speaker_identification_hybrid import (
    detect_speaker_hybrid,
    _contextual_arbiter,
    _detect_from_regex,
    _detect_from_ner
)

@pytest.mark.asyncio
class TestSpeakerIdentificationHybrid:

    @patch("utils.speaker_identification_hybrid.llm_mini")
    async def test_contextual_arbiter_yes(self, mock_llm):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = "YES"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await _contextual_arbiter("Hi, I am John.", "John")
        assert result is True
        mock_llm.ainvoke.assert_called_once()

    @patch("utils.speaker_identification_hybrid.llm_mini")
    async def test_contextual_arbiter_no(self, mock_llm):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = "NO"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await _contextual_arbiter("I know a guy named John.", "John")
        assert result is False

    def test_detect_from_regex_positive(self):
        assert _detect_from_regex("My name is Alice", "en") == "Alice"
        assert _detect_from_regex("我是李华", "zh") == "李华"
        assert _detect_from_regex("Ich bin Hans", "de") == "Hans"

    def test_detect_from_regex_negative(self):
        assert _detect_from_regex("The weather is nice", "en") is None
        assert _detect_from_regex("I am happy", "en") is None # 'happy' is not capitalized in regex pattern

    @patch("utils.speaker_identification_hybrid.llm_mini")
    async def test_detect_from_ner_positive(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "Bob"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await _detect_from_ner("Let me introduce myself, I'm Bob.", "en")
        assert result == "Bob"

    @patch("utils.speaker_identification_hybrid.llm_mini")
    async def test_detect_speaker_hybrid_full_flow(self, mock_llm):
        # Mock for Stage 3 Arbiter (Stage 1 will match 'Alice' via regex)
        mock_response = MagicMock()
        mock_response.content = "YES"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await detect_speaker_hybrid("Hello everyone, my name is Alice.", "en")
        assert result == "Alice"

    @patch("utils.speaker_identification_hybrid.llm_mini")
    async def test_detect_speaker_hybrid_failed_arbiter(self, mock_llm):
        # Stage 1 matches 'John', but Stage 3 says NO
        mock_response = MagicMock()
        mock_response.content = "NO"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await detect_speaker_hybrid("His name is John.", "en")
        assert result is None
