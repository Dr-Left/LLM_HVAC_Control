import re


def parse_output(output_str, n_rooms):
    pattern = r"\[([\-0-9\s,]+)\]"
    matches = re.finditer(pattern, output_str)

    for match in matches:
        try:
            numbers_str = match.group(1)
            numbers = [int(num.strip()) for num in numbers_str.split(",")]
            if len(numbers) == n_rooms:
                return numbers
        except Exception as e:
            continue

    print(f"No valid matches found with {n_rooms} numbers.")
    return None
