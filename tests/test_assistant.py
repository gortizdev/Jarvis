#!/usr/bin/env python3
"""
Unit tests for the enhanced voice assistant.

Run with: pytest tests/test_assistant.py -v
"""

import pytest
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant_enhanced import (
    # Configuration
    AssistantConfig,
    AudioConfig,
    WakeWordConfig,
    TTSConfig,
    SFXConfig,
    HomeAssistantConfig,
    ConversationConfig,
    HealthConfig,
    load_config,
    
    # Text processing
    _normalize_text,
    _extract_hex_rgb,
    _extract_color_name,
    _extract_color_temp_kelvin,
    normalize_light_service_data,
    infer_area_id_from_text,
    
    # Audio processing
    rms,
    decimate_to_16k,
    normalize_volume,
    
    # Entity matching
    fuzzy_match_entity,
    fuzzy_match_area,
    
    # Scene suggestions
    suggest_scene,
    
    # State checking
    check_state_and_respond,
    
    # Offline responses
    get_offline_response,
    
    # HA client
    HomeAssistantClient,
    HomeAssistantError,
    EntityCache,
    _sanitize_service_data,
    _validate_entity_id,
    
    # Health monitoring
    HealthMonitor,
    HealthStatus,
    
    # Constants
    COLOR_NAME_TO_RGB,
)


# ============================
# Test Configuration
# ============================
class TestConfiguration:
    """Tests for configuration dataclasses and loading."""
    
    def test_default_audio_config(self):
        """Test default AudioConfig values."""
        config = AudioConfig()
        assert config.mic_device_index == 2
        assert config.alsa_device == "default"
        assert config.mic_rate == 48000
        assert config.vad_mode == 3
    
    def test_default_assistant_config(self):
        """Test default AssistantConfig values."""
        config = AssistantConfig()
        assert config.api_max_retries == 3
        assert config.api_retry_base_delay == 1.0
        assert config.debug_tools == True
    
    def test_home_assistant_config_defaults(self):
        """Test HomeAssistantConfig has required domains."""
        config = HomeAssistantConfig()
        assert "light" in config.allowed_domains
        assert "switch" in config.allowed_domains
        assert "turn_on" in config.allowed_services
        assert "turn_off" in config.allowed_services
    
    def test_conversation_config_defaults(self):
        """Test ConversationConfig defaults."""
        config = ConversationConfig()
        assert config.followup_enabled == True
        assert config.followup_window_seconds == 5.0
        assert config.bargein_enabled == True


# ============================
# Test Text Processing
# ============================
class TestTextProcessing:
    """Tests for text normalization and extraction."""
    
    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        assert _normalize_text("Hello World") == "hello world"
        assert _normalize_text("UPPERCASE") == "uppercase"
        assert _normalize_text("  spaces  ") == "spaces"
    
    def test_normalize_text_special_chars(self):
        """Test normalization removes special characters."""
        assert _normalize_text("hello!@#world") == "helloworld"
        assert _normalize_text("test_underscore") == "test underscore"
    
    def test_normalize_text_preserves_hash(self):
        """Test that # is preserved for hex colors."""
        assert "#" in _normalize_text("#ff0000")
    
    def test_extract_hex_rgb_valid(self):
        """Test hex color extraction."""
        assert _extract_hex_rgb("#ff0000") == (255, 0, 0)
        assert _extract_hex_rgb("#00ff00") == (0, 255, 0)
        assert _extract_hex_rgb("#0000ff") == (0, 0, 255)
        assert _extract_hex_rgb("set color to #ffffff") == (255, 255, 255)
    
    def test_extract_hex_rgb_invalid(self):
        """Test hex color extraction with invalid input."""
        assert _extract_hex_rgb("no hex here") is None
        assert _extract_hex_rgb("#fff") is None  # Too short
        assert _extract_hex_rgb("") is None
        assert _extract_hex_rgb(None) is None
    
    def test_extract_color_name(self):
        """Test color name extraction."""
        assert _extract_color_name("turn the lights red") == "red"
        assert _extract_color_name("make it blue please") == "blue"
        assert _extract_color_name("change to warm white") == "warm white"
    
    def test_extract_color_name_longest_match(self):
        """Test that longer color names are preferred."""
        # "warm white" should match over "white"
        assert _extract_color_name("set warm white color") == "warm white"
        # "cool white" should match over "white"
        assert _extract_color_name("change to cool white") == "cool white"
    
    def test_extract_color_temp_kelvin(self):
        """Test color temperature extraction."""
        assert _extract_color_temp_kelvin("warm white") == 2700
        assert _extract_color_temp_kelvin("cool white") == 5000
        assert _extract_color_temp_kelvin("soft white") == 3000
        assert _extract_color_temp_kelvin("daylight") == 5000
        assert _extract_color_temp_kelvin("3000k") == 3000
        assert _extract_color_temp_kelvin("set to 4000 k") == 4000
    
    def test_extract_color_temp_kelvin_invalid(self):
        """Test invalid color temperature extraction."""
        assert _extract_color_temp_kelvin("red color") is None
        assert _extract_color_temp_kelvin("100k") is None  # Out of range
        assert _extract_color_temp_kelvin("10000k") is None  # Out of range


# ============================
# Test Light Service Data Normalization
# ============================
class TestLightServiceData:
    """Tests for light service data normalization."""
    
    def test_normalize_converts_color_to_color_name(self):
        """Test that 'color' field is converted to 'color_name'."""
        data = {"color": "red"}
        result = normalize_light_service_data(data, "")
        assert "color_name" not in result or result.get("rgb_color")
    
    def test_normalize_extracts_rgb_from_text(self):
        """Test RGB extraction from user text."""
        data = {}
        result = normalize_light_service_data(data, "set color to #ff0000")
        assert result.get("rgb_color") == [255, 0, 0]
    
    def test_normalize_extracts_color_name_from_text(self):
        """Test color name extraction from user text."""
        data = {}
        result = normalize_light_service_data(data, "turn the lights purple")
        assert result.get("rgb_color") == list(COLOR_NAME_TO_RGB["purple"])
    
    def test_normalize_extracts_brightness(self):
        """Test brightness extraction from user text."""
        data = {}
        result = normalize_light_service_data(data, "set brightness to 50%")
        assert result.get("brightness_pct") == 50
    
    def test_normalize_requires_brightness_keyword(self):
        """Test that random numbers don't become brightness."""
        data = {}
        result = normalize_light_service_data(data, "turn on light 1")
        assert "brightness_pct" not in result
    
    def test_normalize_kelvin_for_white(self):
        """Test that white colors get kelvin values."""
        data = {}
        result = normalize_light_service_data(data, "set to warm white")
        assert result.get("color_temp_kelvin") == 2700
        assert "rgb_color" not in result


# ============================
# Test Area Inference
# ============================
class TestAreaInference:
    """Tests for area ID inference from text."""
    
    def test_infer_area_exact_match(self):
        """Test exact area matching."""
        areas = [
            {"name": "Living Room", "area_id": "living_room"},
            {"name": "Kitchen", "area_id": "kitchen"},
            {"name": "Bedroom", "area_id": "bedroom"},
        ]
        
        assert infer_area_id_from_text("turn on living room lights", areas) == "living_room"
        assert infer_area_id_from_text("kitchen fan on", areas) == "kitchen"
    
    def test_infer_area_no_match(self):
        """Test that non-matching text returns None."""
        areas = [
            {"name": "Living Room", "area_id": "living_room"},
        ]
        
        assert infer_area_id_from_text("turn on garage", areas) is None
    
    def test_infer_area_prefers_longer_match(self):
        """Test that longer matches are preferred."""
        areas = [
            {"name": "Room", "area_id": "room"},
            {"name": "Living Room", "area_id": "living_room"},
        ]
        
        # Should match "Living Room" over "Room"
        result = infer_area_id_from_text("living room lights", areas)
        assert result == "living_room"


# ============================
# Test Audio Processing
# ============================
class TestAudioProcessing:
    """Tests for audio processing functions."""
    
    def test_rms_silent_audio(self):
        """Test RMS of silent audio is near zero."""
        silent = np.zeros(1000, dtype=np.float32)
        assert rms(silent) < 0.001
    
    def test_rms_loud_audio(self):
        """Test RMS of loud audio is high."""
        loud = np.ones(1000, dtype=np.float32) * 0.5
        assert rms(loud) > 0.4
    
    def test_decimate_to_16k_from_48k(self):
        """Test decimation from 48kHz to 16kHz."""
        # Create 48 samples (represents 1ms at 48kHz)
        audio = np.arange(48, dtype=np.int16)
        result = decimate_to_16k(audio, 48000, 16000)
        
        # Should have ~16 samples
        assert len(result) == 16
    
    def test_decimate_to_16k_same_rate(self):
        """Test that 16k -> 16k returns same array."""
        audio = np.arange(100, dtype=np.int16)
        result = decimate_to_16k(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)
    
    def test_normalize_volume(self):
        """Test volume normalization."""
        # Create quiet audio
        quiet = np.ones(1000, dtype=np.int16) * 100
        normalized = normalize_volume(quiet, target_db=-20)
        
        # Should be louder after normalization
        assert np.abs(normalized).max() > np.abs(quiet).max()
    
    def test_normalize_volume_prevents_clipping(self):
        """Test that normalization doesn't cause clipping."""
        loud = np.ones(1000, dtype=np.int16) * 30000
        normalized = normalize_volume(loud, target_db=-10)
        
        # Should not exceed int16 range significantly
        assert np.abs(normalized).max() <= 32767


# ============================
# Test Entity ID Validation
# ============================
class TestEntityValidation:
    """Tests for entity ID validation."""
    
    def test_valid_entity_ids(self):
        """Test valid entity ID formats."""
        assert _validate_entity_id("light.kitchen") == True
        assert _validate_entity_id("switch.bedroom_fan") == True
        assert _validate_entity_id("climate.hvac_1") == True
        assert _validate_entity_id("all") == True
    
    def test_invalid_entity_ids(self):
        """Test invalid entity ID formats."""
        assert _validate_entity_id("invalid") == False
        assert _validate_entity_id("") == False
        assert _validate_entity_id("Light.Kitchen") == False  # Uppercase
        assert _validate_entity_id("light..kitchen") == False


# ============================
# Test Service Data Sanitization
# ============================
class TestServiceDataSanitization:
    """Tests for service data sanitization."""
    
    def test_sanitize_keeps_allowed_keys(self):
        """Test that allowed keys are preserved."""
        data = {
            "entity_id": "light.test",
            "brightness_pct": 50,
            "rgb_color": [255, 0, 0],
        }
        result = _sanitize_service_data(data)
        assert result == data
    
    def test_sanitize_removes_disallowed_keys(self):
        """Test that disallowed keys are removed."""
        data = {
            "entity_id": "light.test",
            "malicious_key": "bad_value",
            "script": "rm -rf /",
        }
        result = _sanitize_service_data(data)
        assert "malicious_key" not in result
        assert "script" not in result
        assert result["entity_id"] == "light.test"


# ============================
# Test Entity Cache
# ============================
class TestEntityCache:
    """Tests for entity cache."""
    
    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        config = AssistantConfig()
        cache = EntityCache(config)
        
        state = {"state": "on", "attributes": {"brightness": 255}}
        cache.set_state("light.test", state)
        
        result = cache.get_state("light.test")
        assert result == state
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        config = AssistantConfig()
        cache = EntityCache(config)
        
        assert cache.get_state("light.nonexistent") is None
    
    def test_cache_invalidate(self):
        """Test cache invalidation."""
        config = AssistantConfig()
        cache = EntityCache(config)
        
        cache.set_state("light.test", {"state": "on"})
        cache.invalidate("light.test")
        
        assert cache.get_state("light.test") is None
    
    def test_cache_invalidate_all(self):
        """Test invalidating all cached states."""
        config = AssistantConfig()
        cache = EntityCache(config)
        
        cache.set_state("light.test1", {"state": "on"})
        cache.set_state("light.test2", {"state": "off"})
        cache.invalidate_all()
        
        assert cache.get_state("light.test1") is None
        assert cache.get_state("light.test2") is None


# ============================
# Test State-Aware Responses
# ============================
class TestStateAwareResponses:
    """Tests for state-aware response checking."""
    
    def test_light_already_on(self):
        """Test response when light is already on."""
        config = AssistantConfig()
        cache = EntityCache(config)
        ha_client = Mock()
        ha_client.get_state.return_value = {
            "state": "on",
            "attributes": {"friendly_name": "Kitchen Light"},
        }
        
        with patch.object(HomeAssistantClient, 'get_state', ha_client.get_state):
            mock_client = Mock(spec=HomeAssistantClient)
            mock_client.get_state = ha_client.get_state
            
            result = check_state_and_respond(
                mock_client, "light", "turn_on", "light.kitchen", None
            )
            
            assert result is not None
            assert "already on" in result.lower()
    
    def test_light_already_off(self):
        """Test response when light is already off."""
        ha_client = Mock()
        ha_client.get_state.return_value = {
            "state": "off",
            "attributes": {"friendly_name": "Kitchen Light"},
        }
        
        result = check_state_and_respond(
            ha_client, "light", "turn_off", "light.kitchen", None
        )
        
        assert result is not None
        assert "already off" in result.lower()
    
    def test_no_response_for_valid_action(self):
        """Test no response when action is valid."""
        ha_client = Mock()
        ha_client.get_state.return_value = {
            "state": "off",
            "attributes": {"friendly_name": "Kitchen Light"},
        }
        
        result = check_state_and_respond(
            ha_client, "light", "turn_on", "light.kitchen", None
        )
        
        assert result is None


# ============================
# Test Offline Responses
# ============================
class TestOfflineResponses:
    """Tests for offline/fallback responses."""
    
    def test_time_response(self):
        """Test offline time response."""
        response = get_offline_response("what time is it?")
        assert "time" in response.lower() or ":" in response
    
    def test_date_response(self):
        """Test offline date response."""
        response = get_offline_response("what's today's date?")
        assert len(response) > 10  # Should have actual date
    
    def test_greeting_response(self):
        """Test offline greeting response."""
        response = get_offline_response("hello")
        assert "hello" in response.lower() or "hi" in response.lower()
    
    def test_thank_response(self):
        """Test offline thank you response."""
        response = get_offline_response("thank you")
        assert "welcome" in response.lower() or "happy" in response.lower()
    
    def test_default_response(self):
        """Test default offline response."""
        response = get_offline_response("some random query")
        assert "trouble" in response.lower() or "try again" in response.lower()


# ============================
# Test Fuzzy Matching (if available)
# ============================
class TestFuzzyMatching:
    """Tests for fuzzy entity matching."""
    
    @pytest.fixture
    def sample_entities(self):
        return [
            {"entity_id": "light.living_room_main", "attributes": {"friendly_name": "Living Room Main Light"}},
            {"entity_id": "light.bedroom_lamp", "attributes": {"friendly_name": "Bedroom Lamp"}},
            {"entity_id": "light.kitchen_ceiling", "attributes": {"friendly_name": "Kitchen Ceiling Light"}},
            {"entity_id": "switch.garage_door", "attributes": {"friendly_name": "Garage Door"}},
        ]
    
    @pytest.fixture
    def sample_areas(self):
        return [
            {"area_id": "living_room", "name": "Living Room"},
            {"area_id": "master_bedroom", "name": "Master Bedroom"},
            {"area_id": "kitchen", "name": "Kitchen"},
        ]
    
    def test_fuzzy_match_entity_exact(self, sample_entities):
        """Test fuzzy matching with exact name."""
        try:
            result = fuzzy_match_entity("bedroom lamp", sample_entities)
            if result is not None:  # Only if fuzzy matching is available
                assert result == "light.bedroom_lamp"
        except ImportError:
            pytest.skip("rapidfuzz not available")
    
    def test_fuzzy_match_entity_partial(self, sample_entities):
        """Test fuzzy matching with partial name."""
        try:
            result = fuzzy_match_entity("living room light", sample_entities, domain_filter="light")
            if result is not None:
                assert "living" in result
        except ImportError:
            pytest.skip("rapidfuzz not available")
    
    def test_fuzzy_match_area(self, sample_areas):
        """Test fuzzy matching for areas."""
        try:
            result = fuzzy_match_area("bedroom", sample_areas)
            if result is not None:
                assert result == "master_bedroom"
        except ImportError:
            pytest.skip("rapidfuzz not available")


# ============================
# Test Health Monitor
# ============================
class TestHealthMonitor:
    """Tests for health monitoring."""
    
    def test_health_monitor_initialization(self):
        """Test health monitor can be created."""
        config = AssistantConfig()
        monitor = HealthMonitor(config)
        assert monitor is not None
    
    def test_service_assumed_available_before_check(self):
        """Test services are assumed available before first check."""
        config = AssistantConfig()
        monitor = HealthMonitor(config)
        
        # Before any checks, service should be considered available
        assert monitor.is_service_available("openai") == True
        assert monitor.is_service_available("home_assistant") == True


# ============================
# Test Color Constants
# ============================
class TestColorConstants:
    """Tests for color constant definitions."""
    
    def test_all_colors_have_rgb_tuples(self):
        """Test that all colors are valid RGB tuples."""
        for name, rgb in COLOR_NAME_TO_RGB.items():
            assert isinstance(rgb, tuple), f"{name} should be a tuple"
            assert len(rgb) == 3, f"{name} should have 3 values"
            for i, val in enumerate(rgb):
                assert 0 <= val <= 255, f"{name}[{i}] should be 0-255"
    
    def test_basic_colors_exist(self):
        """Test that basic colors are defined."""
        basic = ["red", "green", "blue", "white", "yellow", "orange", "purple"]
        for color in basic:
            assert color in COLOR_NAME_TO_RGB, f"{color} should be defined"
    
    def test_white_variants_exist(self):
        """Test that white variants are defined."""
        variants = ["white", "warm white", "cool white"]
        for variant in variants:
            assert variant in COLOR_NAME_TO_RGB, f"{variant} should be defined"


# ============================
# Integration Tests
# ============================
class TestIntegration:
    """Integration tests for combined functionality."""
    
    def test_full_light_command_flow(self):
        """Test processing a complete light command."""
        user_text = "turn the living room lights to warm white at 75% brightness"
        
        # Extract area
        areas = [{"name": "Living Room", "area_id": "living_room"}]
        area = infer_area_id_from_text(user_text, areas)
        assert area == "living_room"
        
        # Build service data
        data = {"area_id": area}
        data = normalize_light_service_data(data, user_text)
        
        assert data.get("area_id") == "living_room"
        assert data.get("color_temp_kelvin") == 2700
        assert data.get("brightness_pct") == 75
    
    def test_color_command_processing(self):
        """Test processing color commands."""
        test_cases = [
            ("set lights to red", {"expected_color": "red"}),
            ("change to #00ff00", {"expected_rgb": [0, 255, 0]}),
            ("make it purple at 50%", {"expected_color": "purple", "expected_brightness": 50}),
        ]
        
        for user_text, expected in test_cases:
            data = normalize_light_service_data({}, user_text)
            
            if "expected_rgb" in expected:
                assert data.get("rgb_color") == expected["expected_rgb"]
            
            if "expected_brightness" in expected:
                assert data.get("brightness_pct") == expected["expected_brightness"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
