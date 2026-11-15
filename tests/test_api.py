"""API tests with isolated SQLite and mocked speech/LLM services."""

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import unquote
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.core.reservation_policy import restaurant_today
from src.api.main import create_app
from src.api.orchestration import APIBackend
from src.database.connection import create_connection
from src.database.init_db import init_database
from src.services.agent_service import AgentService
from src.services.llm_service import OllamaConnectionError, OllamaService
from src.services.stt_service import EmptyTranscriptionError, InvalidAudioError, STTModelError, STTService
from src.services.tts_service import TTSSynthesisError


@pytest.fixture
def setup(tmp_path):
    db = tmp_path / "restaurant.db"
    init_database(db_path=db, seed=True)
    # Synthetic test prices, not restaurant-approved seed or live prices.
    with create_connection(db) as conn:
        conn.executemany("UPDATE menu_items SET price = ? WHERE name = ?", [
            (449, 'Caprese Salad'), (899, 'Grilled Salmon Fillet'),
            (299, 'Classic Tiramisu'), (149, 'Fresh Mint Lemonade'),
        ])
    llm = MagicMock(spec=OllamaService)
    llm.chat.side_effect = OllamaConnectionError("offline")
    llm.generate_restaurant_reply.side_effect = OllamaConnectionError("offline")
    stt, tts = MagicMock(), MagicMock()
    backend = APIBackend(db, 8, agent=AgentService(db_path=str(db), llm_service=llm), stt=stt, tts=tts)
    app = create_app(backend)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield SimpleNamespace(db=db, client=client, backend=backend, stt=stt, tts=tts, tmp=tmp_path)


def test_health_checks_database_without_loading_models(setup):
    res = setup.client.get("/api/health")
    assert res.status_code == 200 and res.json() == {"status": "ok", "database": "ok"}
    setup.stt.assert_not_called()
    setup.tts.assert_not_called()
    setup.backend.db_path = setup.tmp / "absent.db"
    assert setup.client.get("/api/health").status_code == 503
    assert not setup.backend.db_path.exists()


def test_chat_response_session_continuity_and_isolation(setup):
    first, second = str(uuid4()), str(uuid4())
    result = setup.client.post("/api/chat", json={"message": "I want to book a table", "session_id": first})
    assert result.status_code == 200
    data = result.json()
    assert data["session_id"] == first
    assert data["intent"] == "reservation_request"
    assert data["needs_clarification"] is True
    assert {"text", "intent", "entities", "data", "success", "is_fallback", "needs_clarification", "missing_slots"} <= data.keys()
    tomorrow = (restaurant_today() + timedelta(days=1)).isoformat()
    followup = setup.client.post("/api/chat", json={"message": f"For {tomorrow}", "session_id": first})
    assert followup.status_code == 200
    assert setup.backend.agent.get_session(first).slots["requested_reservation_date"] == tomorrow
    other = setup.client.post("/api/chat", json={"message": "I want to book a table", "session_id": second})
    assert other.status_code == 200
    assert "requested_reservation_date" not in setup.backend.agent.get_session(second).slots
    assert setup.backend.agent.get_session(first).slots["requested_reservation_date"] == tomorrow


def test_chat_validation_does_not_reflect_sensitive_input(setup):
    assert setup.client.post("/api/chat", json={"message": "hello"}).status_code == 422
    assert setup.client.post("/api/chat", json={"message": " ", "session_id": str(uuid4())}).status_code == 422
    response = setup.client.post("/api/chat", json={"message": "secret", "session_id": "bad"})
    assert response.status_code == 422 and "secret" not in response.text


def test_menu_and_filtering(setup):
    all_items = setup.client.get("/api/menu").json()
    assert all_items["categories"] and all_items["items"]
    category = all_items["categories"][0]
    filtered = setup.client.get("/api/menu", params={"category": category}).json()
    assert filtered["items"] and all(item["category"] == category for item in filtered["items"])
    assert setup.client.get("/api/menu", params={"category": "not a category"}).json()["items"] == []


def test_menu_returns_database_prices(setup):
    with create_connection(setup.db) as conn:
        expected = {row['name']: row['price'] for row in conn.execute(
            'SELECT name, price FROM menu_items WHERE is_available = 1')}
    items = setup.client.get('/api/menu').json()['items']
    assert {item['name']: item['price'] for item in items} == expected
    filtered = setup.client.get('/api/menu', params={'category': 'Appetizers'}).json()['items']
    assert {item['name']: item['price'] for item in filtered} == {
        name: expected[name] for name in ('Bruschetta Classica', 'Crispy Calamari', 'Caprese Salad')
    }


@pytest.mark.parametrize('message,expected_name,expected_price', [
    ('How much does the Caprese Salad cost?', 'Caprese Salad', 449),
    ('How much is Grilled Salmon Fillet?', 'Grilled Salmon Fillet', 899),
    ('How much is Classic Tiramisu?', 'Classic Tiramisu', 299),
    ('How much is Fresh Mint Lemonade?', 'Fresh Mint Lemonade', 149),
])
def test_chat_menu_prices_use_rupees(setup, message, expected_name, expected_price):
    response = setup.client.post('/api/chat', json={'message': message, 'session_id': str(uuid4())})
    assert response.status_code == 200
    result = response.json()
    assert result['data']['name'] == expected_name and result['data']['price'] == expected_price
    assert f'₹{expected_price}' in result['text']
    assert '$' not in result['text'] and 'dollar' not in result['text'].lower()


def booking(**overrides):
    values = {"customer_name": "Alice", "customer_phone": "9870001234", "party_size": 2,
              "reservation_date": (restaurant_today() + timedelta(days=1)).isoformat(), "reservation_time": "18:00"}
    values.update(overrides)
    return values


def test_reservation_success_is_committed(setup):
    response = setup.client.post("/api/reservations", json=booking())
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "confirmed"
    conn = create_connection(setup.db)
    try:
        assert conn.execute("SELECT customer_name FROM reservations WHERE id=?", (data["reservation_id"],)).fetchone()[0] == "Alice"
    finally:
        conn.close()


def test_invalid_reservation_and_unavailability(setup):
    assert setup.client.post("/api/reservations", json=booking(customer_name=" ")).status_code == 422
    assert setup.client.post("/api/reservations", json=booking(party_size=-1)).status_code == 422
    assert setup.client.post("/api/reservations", json=booking(party_size=True)).status_code == 422
    assert setup.client.post("/api/reservations", json=booking(party_size=999)).status_code == 409
    assert setup.client.post("/api/reservations", json=booking(reservation_time="05:00")).status_code == 422


def test_commit_failure_never_returns_confirmation(setup):
    with patch("src.api.orchestration.get_db") as manager:
        manager.return_value.__enter__.return_value = MagicMock()
        manager.return_value.__exit__.side_effect = RuntimeError("secret database path")
        with patch("src.api.orchestration.create_reservation", return_value={"reservation_id": 1}):
            response = setup.client.post("/api/reservations", json=booking())
    assert response.status_code == 500
    assert "secret" not in response.text and "reservation_id" not in response.text


def upload(client, payload=b"audio", content_type="audio/wav", sid=None):
    return client.post("/api/voice", data={"session_id": sid or str(uuid4())}, files={"file": ("test.wav", payload, content_type)})


def test_voice_success_and_temp_cleanup(setup):
    paths = []
    def transcribe(path, initial_prompt=None):
        paths.append(path)
        assert path.read_bytes() == b"audio"
        return "hello café"
    def synthesize(text, path):
        paths.append(path)
        path.write_bytes(b"RIFFmockWAVE")
        return path
    setup.stt.transcribe.side_effect = transcribe
    setup.tts.synthesize.side_effect = synthesize
    sid = str(uuid4())
    response = upload(setup.client, sid=sid)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFFmockWAVE"
    assert unquote(response.headers["x-transcript"]) == "hello café"
    assert unquote(response.headers["x-response-text"])
    assert response.headers["x-session-id"] == sid
    assert all(not path.exists() and not path.parent.exists() for path in paths)
    assert response.headers["cache-control"] == "no-store"


def test_voice_displays_rupee_symbol_and_speaks_rupees(setup):
    setup.stt.transcribe.return_value = 'How much is Caprese Salad?'
    setup.tts.synthesize.side_effect = lambda text, path: path.write_bytes(b'wav')
    response = upload(setup.client)
    assert response.status_code == 200
    assert '₹449' in unquote(response.headers['x-response-text'])
    spoken_text = setup.tts.synthesize.call_args.args[0]
    assert '449 rupees' in spoken_text and '₹' not in spoken_text and '$' not in spoken_text


def test_voice_prompt_uses_available_live_menu_only(setup):
    with create_connection(setup.db) as conn:
        conn.execute("UPDATE menu_items SET is_available = 0 WHERE name = ?", ("Caprese Salad",))
    setup.stt.transcribe.return_value = 'hello'
    setup.tts.synthesize.side_effect = lambda text, path: path.write_bytes(b'wav')
    response = upload(setup.client)
    assert response.status_code == 200
    prompt = setup.stt.transcribe.call_args.kwargs['initial_prompt']
    assert prompt.startswith('Restaurant menu items: ') and len(prompt) <= 400
    assert 'Bruschetta Classica' in prompt and 'Caprese Salad' not in prompt
    assert 'Mediterranean Chicken Burger' not in prompt


@pytest.mark.parametrize('reason', ['empty', 'database_error', 'missing'])
def test_voice_prompt_falls_back_without_menu(setup, reason, tmp_path):
    if reason == 'empty':
        with create_connection(setup.db) as conn:
            conn.execute('UPDATE menu_items SET is_available = 0')
    elif reason == 'missing':
        setup.backend.db_path = tmp_path / 'missing.db'
    setup.stt.transcribe.return_value = 'hello'
    setup.tts.synthesize.side_effect = lambda text, path: path.write_bytes(b'wav')
    # Isolate the transcription fallback from the separate chat database access.
    with patch.object(setup.backend, 'chat', return_value={'text': 'hello'}):
        if reason == 'database_error':
            import sqlite3
            with patch('src.api.orchestration.get_menu_items', side_effect=sqlite3.OperationalError('offline')):
                response = upload(setup.client)
        else:
            response = upload(setup.client)
    assert response.status_code == 200
    assert setup.stt.transcribe.call_args.kwargs['initial_prompt'] is None
    if reason == 'missing':
        assert not setup.backend.db_path.exists()


def test_menu_prompt_is_bounded_without_truncating_names(setup):
    items = [{'name': 'x' * 390}, {'name': 'Caprese Salad'}, {'name': 'y' * 390}]
    with patch('src.api.orchestration.get_menu_items', return_value=items):
        prompt = setup.backend._menu_vocabulary_prompt()
    assert prompt == 'Restaurant menu items: Caprese Salad'
    assert len(prompt) <= 400


def test_voice_validation(setup):
    assert setup.client.post("/api/voice", data={"session_id": str(uuid4())}).status_code == 422
    assert setup.client.post("/api/voice", files={"file": ("a.wav", b"audio", "audio/wav")}).status_code == 422
    assert upload(setup.client, payload=b"").status_code == 400
    assert upload(setup.client, payload=b"123456789").status_code == 413
    assert upload(setup.client, content_type="text/plain").status_code == 415
    assert upload(setup.client, sid="invalid").status_code == 422
    setup.stt.transcribe.assert_not_called()


@pytest.mark.parametrize("error,status", [
    (InvalidAudioError("private path"), 422),
    (EmptyTranscriptionError("private path"), 422),
    (STTModelError("private path"), 503),
    (RuntimeError("private path"), 500),
])
def test_voice_stt_errors_and_cleanup(setup, error, status):
    paths = []
    def fail(path, initial_prompt=None):
        paths.append(path)
        raise error
    setup.stt.transcribe.side_effect = fail
    response = upload(setup.client)
    assert response.status_code == status
    assert "private" not in response.text
    assert paths and all(not p.exists() and not p.parent.exists() for p in paths)
    setup.tts.synthesize.assert_not_called()


def test_real_invalid_audio_is_rejected_without_loading_whisper(setup):
    setup.backend.stt = STTService()
    with patch("src.services.stt_service.WhisperModel") as model:
        response = upload(setup.client, payload=b"bad")
    assert response.status_code == 422
    model.assert_not_called()
    setup.tts.synthesize.assert_not_called()


def test_voice_tts_error_and_cleanup(setup):
    paths = []
    def transcribe(path, initial_prompt=None):
        paths.append(path)
        return "hello"
    def synthesize(text, path):
        paths.append(path)
        path.write_bytes(b"partial")
        raise TTSSynthesisError("private path")
    setup.stt.transcribe.side_effect = transcribe
    setup.tts.synthesize.side_effect = synthesize
    response = upload(setup.client)
    assert response.status_code == 503 and "private" not in response.text
    assert len(paths) == 2 and all(not p.exists() and not p.parent.exists() for p in paths)


def test_voice_shares_chat_session_state(setup):
    sid = str(uuid4())
    assert setup.client.post("/api/chat", json={"message": "I want to book a table", "session_id": sid}).status_code == 200
    tomorrow = (restaurant_today() + timedelta(days=1)).isoformat()
    setup.stt.transcribe.return_value = f"For {tomorrow}"
    setup.tts.synthesize.side_effect = lambda text, path: path.write_bytes(b"wav")
    assert upload(setup.client, sid=sid).status_code == 200
    assert setup.backend.agent.get_session(sid).slots["requested_reservation_date"] == tomorrow


def test_voice_limit_is_inclusive_and_cleans_up_on_oversize(setup):
    from tempfile import TemporaryDirectory
    directories = []
    def tracked_directory(*args, **kwargs):
        instance = TemporaryDirectory(*args, **kwargs)
        directories.append(Path(instance.name))
        return instance
    setup.stt.transcribe.return_value = "hello"
    setup.tts.synthesize.side_effect = lambda text, path: path.write_bytes(b"wav")
    with patch("src.api.orchestration.tempfile.TemporaryDirectory", side_effect=tracked_directory):
        assert upload(setup.client, payload=b"12345678").status_code == 200
        assert upload(setup.client, payload=b"123456789").status_code == 413
    assert len(directories) == 2 and all(not path.exists() for path in directories)


def test_cors_exposes_voice_headers(setup):
    response = setup.client.options("/api/voice", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    bad = setup.client.options("/api/voice", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_slots_and_reservation_policy_are_consistent(setup, monkeypatch):
    from src.core.config import config
    monkeypatch.setattr(config, "opening_time", "10:00")
    monkeypatch.setattr(config, "closing_time", "23:00")
    monkeypatch.setattr(config, "default_reservation_duration_minutes", 90)
    day = (restaurant_today() + timedelta(days=1)).isoformat()
    slots = setup.client.get('/api/reservations/slots', params={'date': day})
    assert slots.status_code == 200
    assert slots.json()['slots'][0] == '10:00' and slots.json()['slots'][-1] == '21:30'
    assert setup.client.get('/api/reservations/slots', params={'date': 'bad'}).status_code == 422
    for fields in ({'customer_phone': '98765-43210'}, {'reservation_time': '18:15'},
                   {'reservation_time': '22:30'}, {'reservation_date': (restaurant_today() - timedelta(days=1)).isoformat()}):
        response = setup.client.post('/api/reservations', json=booking(**fields))
        assert response.status_code == 422
        assert 'mobile' in response.json()['detail'] or 'reservation time' in response.json()['detail']
    assert setup.client.post('/api/reservations', json=booking(customer_phone='+919876543210')).status_code == 201


def test_chat_does_not_accept_llm_normalized_invalid_phone(setup):
    from src.services.agent_service import Intent
    session = setup.backend.agent.get_session('phone-policy')
    session.pending_intent = Intent.RESERVATION_REQUEST
    session.slots = {'requested_reservation_date': (restaurant_today() + timedelta(days=1)).isoformat(),
                     'requested_reservation_time': '18:00', 'number_of_guests': 2,
                     'customer_name': 'Alice', 'customer_phone': None}
    setup.backend.agent.llm.chat.side_effect = None
    setup.backend.agent.llm.chat.return_value = SimpleNamespace(content='{"intent":"reservation_request","customer_phone":"9876543210"}')
    result = setup.backend.agent.handle_message('Phone is 98765-43210', session_id='phone-policy')
    assert result.needs_clarification and 'Indian mobile' in result.text
    assert session.slots['customer_phone'] is None


def test_slots_and_booking_use_kolkata_clock(setup, monkeypatch):
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    from src.core import reservation_policy as policy
    from src.core.config import config
    monkeypatch.setattr(config, 'opening_time', '10:00')
    monkeypatch.setattr(config, 'closing_time', '23:00')
    monkeypatch.setattr(config, 'default_reservation_duration_minutes', 90)
    # 04:45 UTC is 10:15 IST; the first bookable start is 10:30.
    instant = datetime(2030, 1, 1, 4, 45, tzinfo=timezone.utc)
    monkeypatch.setattr(policy, 'restaurant_now', lambda: instant.astimezone(ZoneInfo('Asia/Kolkata')))
    slots = setup.client.get('/api/reservations/slots', params={'date': '2030-01-01'}).json()['slots']
    assert slots[0] == '10:30' and slots[-1] == '21:30'
    assert setup.client.post('/api/reservations', json=booking(reservation_date='2030-01-01', reservation_time='10:00')).status_code == 422
    assert setup.client.post('/api/reservations', json=booking(reservation_date='2030-01-01', reservation_time='10:30')).status_code == 201


def test_chat_menu_follow_up_keeps_session_context(setup):
    session_id = str(uuid4())
    first = setup.client.post('/api/chat', json={'session_id': session_id, 'message': 'How much is Margherita Pizza?'})
    assert first.status_code == 200
    followup = setup.client.post('/api/chat', json={'session_id': session_id, 'message': 'Does it contain dairy?'})
    assert followup.status_code == 200
    assert 'contains Dairy' in followup.json()['text']
    assert followup.json()['entities']['menu_item_name'] == 'Margherita Pizza'


def test_voice_silence_is_rejected_without_loading_model(setup, tmp_path):
    import wave
    path = tmp_path / 'silence.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\x00\x00' * 1600)
    setup.backend.max_upload_bytes = 65536
    setup.backend.stt = STTService()
    with patch('src.services.stt_service.WhisperModel') as model:
        response = upload(setup.client, payload=path.read_bytes())
    assert response.status_code == 422
    model.assert_not_called()
    setup.tts.synthesize.assert_not_called()


def test_voice_real_decoding_preserves_api_and_piper_contract(setup, tmp_path):
    import struct
    import wave
    path = tmp_path / 'sample.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(struct.pack('<hh', 10000, 10000) * 4800)
    setup.backend.max_upload_bytes = 65536
    setup.backend.stt = STTService(model_name='base.en', beam_size=3, vad_filter=True)
    setup.tts.synthesize.side_effect = lambda text, output: output.write_bytes(b'RIFFmockWAVE')
    model = MagicMock()
    model.transcribe.return_value = (iter([SimpleNamespace(text='Tell me about Caprese Salad')]), None)
    with patch.object(setup.backend.stt, '_get_model', return_value=model):
        response = upload(setup.client, payload=path.read_bytes())
    assert response.status_code == 200 and response.content == b'RIFFmockWAVE'
    assert unquote(response.headers['x-transcript']) == 'Tell me about Caprese Salad'
    audio = model.transcribe.call_args.args[0]
    assert audio.ndim == 1 and len(audio) == 1600
    assert model.transcribe.call_args.kwargs == {'language': 'en', 'task': 'transcribe', 'beam_size': 3, 'vad_filter': True, 'initial_prompt': setup.backend._menu_vocabulary_prompt()}
    setup.tts.synthesize.assert_called_once()


def test_chat_booking_confirmation_uses_committed_details_not_llm_greeting(setup):
    from src.services.agent_service import Intent
    session_id = str(uuid4())
    day = (restaurant_today() + timedelta(days=1)).isoformat()
    session = setup.backend.agent.get_session(session_id)
    session.pending_intent = Intent.RESERVATION_REQUEST
    session.slots = {'requested_reservation_date': day, 'requested_reservation_time': '18:00',
                     'number_of_guests': 2, 'customer_name': 'Alice', 'customer_phone': None}
    llm = setup.backend.agent.llm
    llm.chat.side_effect = None
    llm.chat.return_value = SimpleNamespace(content='{"intent":"greeting","reservation_identifier":9876543210}')
    llm.generate_restaurant_reply.side_effect = None
    llm.generate_restaurant_reply.return_value = SimpleNamespace(content='Hello! How can I help you?')

    response = setup.client.post('/api/chat', json={'session_id': session_id, 'message': 'My phone number is +919876543210'})
    assert response.status_code == 200
    data = response.json()
    booking = data['data']
    assert data['success'] and booking['status'] == 'confirmed'
    assert data['entities']['customer_phone'] == '+919876543210'
    assert data['entities']['reservation_identifier'] is None
    assert 'reservation is confirmed' in data['text'].lower()
    for detail in (str(booking['reservation_id']), day, '18:00', 'Alice', '2 guests', booking['table_number']):
        assert detail in data['text']
    assert '+919876543210' not in data['text']
    llm.generate_restaurant_reply.assert_not_called()
    conn = create_connection(setup.db)
    try:
        row = conn.execute('SELECT customer_phone, status FROM reservations WHERE id = ?', (booking['reservation_id'],)).fetchone()
        assert row['customer_phone'] == '+919876543210' and row['status'] == 'confirmed'
    finally:
        conn.close()


def test_private_lookup_and_confirmed_cancellation(setup):
    created = setup.client.post('/api/reservations', json=booking()).json()
    ref, code = created['reservation_id'], created['access_code']
    credentials = {'reservation_id': ref, 'access_code': code}
    assert len(code) >= 30
    wrong = setup.client.post('/api/reservations/lookup', json={**credentials, 'access_code': 'wrong'})
    missing = setup.client.post('/api/reservations/lookup', json={'reservation_id': ref + 9000, 'access_code': 'wrong'})
    assert wrong.status_code == missing.status_code == 404 and wrong.json() == missing.json()
    found = setup.client.post('/api/reservations/lookup', json=credentials)
    assert found.status_code == 200 and found.json()['status'] == 'confirmed'
    assert 'customer_phone' not in found.json() and 'customer_name' not in found.json()
    assert setup.client.post('/api/reservations/cancel', json={**credentials, 'confirm': False}).status_code == 422
    assert setup.client.post('/api/reservations/cancel', json={**credentials, 'confirm': True}).json()['status'] == 'cancelled'
    assert setup.client.post('/api/reservations/cancel', json={**credentials, 'confirm': True}).json()['status'] == 'cancelled'
    with create_connection(setup.db) as conn:
        assert conn.execute('SELECT status FROM reservations WHERE id=?', (ref,)).fetchone()[0] == 'cancelled'
        digest = conn.execute('SELECT secret_hash FROM reservation_access WHERE reservation_id=?', (ref,)).fetchone()[0]
        assert code not in digest
    assert setup.client.post('/api/reservations', json=booking()).status_code == 201


def test_duplicate_and_validation_and_availability(setup, monkeypatch):
    monkeypatch.setenv('RESERVATION_MAX_ADVANCE_DAYS', '3')
    assert setup.client.post('/api/reservations', json=booking(customer_name='Alice 123')).status_code == 422
    assert setup.client.post('/api/reservations', json=booking(reservation_date=(restaurant_today() + timedelta(days=4)).isoformat())).status_code == 422
    assert setup.client.post('/api/reservations', json=booking()).status_code == 201
    assert setup.client.post('/api/reservations', json=booking(customer_phone='+919870001234')).status_code == 409
    day = (restaurant_today() + timedelta(days=1)).isoformat()
    assert setup.client.get('/api/reservations/slots', params={'date': day, 'party_size': 999}).json()['slots'] == []


def test_legacy_reservations_do_not_gain_guessable_credentials(setup):
    created = setup.client.post('/api/reservations', json=booking()).json()
    with create_connection(setup.db) as conn:
        conn.execute('DELETE FROM reservation_access WHERE reservation_id=?', (created['reservation_id'],))
    reply = setup.client.post('/api/reservations/lookup', json={'reservation_id': created['reservation_id'], 'access_code': created['access_code']})
    assert reply.status_code == 404


def test_cancelled_reservation_frees_the_only_suitable_table(setup):
    day = (restaurant_today() + timedelta(days=1)).isoformat()
    with create_connection(setup.db) as conn:
        conn.execute('UPDATE tables SET is_active = 0 WHERE capacity >= 8 AND table_number != ?', ('T6',))
    before = setup.client.get('/api/reservations/slots', params={'date': day, 'party_size': 8}).json()['slots']
    assert '18:00' in before
    created = setup.client.post('/api/reservations', json=booking(party_size=8)).json()
    during = setup.client.get('/api/reservations/slots', params={'date': day, 'party_size': 8}).json()['slots']
    assert '18:00' not in during
    creds = {'reservation_id': created['reservation_id'], 'access_code': created['access_code'], 'confirm': True}
    assert setup.client.post('/api/reservations/cancel', json=creds).status_code == 200
    after = setup.client.get('/api/reservations/slots', params={'date': day, 'party_size': 8}).json()['slots']
    assert '18:00' in after
