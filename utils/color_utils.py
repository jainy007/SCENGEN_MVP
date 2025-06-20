# === COLOR UTILITY ===
def color_text(text: str, color: str) -> str:
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "cyan": "\033[93m",
        "reset": "\033[0m"
    }
    color_code = colors.get(color.lower(), colors["reset"])
    return f"{color_code}{text}{colors['reset']}"