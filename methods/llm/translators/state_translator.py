import logging

logger = logging.getLogger(__name__)


def translate_state(
    room_num,
    observation,
    target_temp,
    room_names=None,
):
    prompt = """Temperature in each room is as follows:
{}
The external climate conditions:
  Outside Temperature: {:.2f} degrees Celsius
  Global Horizontal Irradiance: {:.2f}
  Ground Temperature: {:.2f} degrees Celsius
  Occupant Power: {:.4f} KW
Target Temperature: {:.2f} degrees Celsius"""

    # Format room temperatures with names if available
    room_temps = []
    for i in range(room_num):
        room_name = room_names[i] if room_names and i < len(room_names) else f"Room {i}"
        room_temps.append(f"{room_name}: {observation[i]:.2f} degrees Celsius")

    prompt = prompt.format(
        "\n".join(room_temps),
        observation[room_num],
        observation[room_num + 1],
        observation[room_num + 1 + room_num],
        observation[room_num + 1 + room_num + 1],
        target_temp,
    )
    return prompt
