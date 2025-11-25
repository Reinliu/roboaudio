# spatial_audio_processor/utils.py
def normalize_command(text: str) -> str:
    text = text.lower().strip()

    # Simple keyword mapping
    if "follow me" in text:
        return "follow me"
    if "turn left" in text:
        return "turn left"
    if "turn right" in text:
        return "turn right"
    if "stop" in text:
        return "stop"

    return text  # fallback raw text


def direction_is_valid(command: str, direction_deg: float) -> bool:
    """
    Spatial rules from takehome spec:
      - follow me: any direction ok
      - stop: always ok
      - turn left: only if speaker on left (180–270)
      - (you can add turn right etc.)
      - optional front range gating example
    """
    command = command.lower()

    if command == "stop":
        return True

    if command == "turn left":
        return 180.0 <= direction_deg <= 270.0

    if command == "turn right":
        return 90.0 <= direction_deg <= 180.0  # example

    if command == "follow me":
        return True

    return False
