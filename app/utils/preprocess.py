from bs4 import BeautifulSoup
import re

def is_html(text: str) -> bool:
    """
    Determine whether the provided text appears to contain HTML content.

    Parameters
    ----------
    text : str
        The input string to evaluate.

    Returns
    -------
    bool
        True if the text contains HTML tags, otherwise False.
    """
    return bool(re.search(r"</?[a-z][\s\S]*>", text.strip(), re.IGNORECASE))


def clean_email(content: str) -> str:
    """
    Clean and normalize an email body.

    Converts HTML content to plain text, removes structural elements, and strips common email signatures such as "Best regards" or "Thank you".

    Parameters
    ----------
    content : str
        The raw email body, which may include HTML markup or plain text.

    Returns
    -------
    str
        The cleaned text with HTML removed, signature trimmed, and whitespace normalized.
    """
    if not content:
        return ""

    text = content.strip()
    if is_html(text):
        soup = BeautifulSoup(text, "html.parser")
        for hr in soup.find_all("hr"):
            hr.decompose()
        text = soup.get_text(separator="\n", strip=True)

    stop_phrases = [
        "best regards", "kind regards", "regards",
        "thank you", "thanks", "cheers", "sincerely",
        "yours truly", "yours faithfully"
    ]

    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        lower_line = line.strip().lower()
        if any(phrase in lower_line for phrase in stop_phrases):
            break 
        clean_lines.append(line)

    cleaned_text = " ".join(clean_lines)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text
