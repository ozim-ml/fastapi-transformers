import re

def join_subject_body(subject: str, body: str) -> str:
    """
    Combine the email subject and body into one string.
    Ensures subject ends with a separator before body content.
    """
    subject = (subject or "").strip()
    body = (body or "").strip()

    if not subject:
        return body

    if re.search(r"[A-Za-z]$", subject):
        subject += ": "
    else:
        subject = re.sub(r"[^A-Za-z]+$", ": ", subject)

    return (subject + body).strip()


def is_html(text: str) -> bool:
    """
    Detect if the given text appears to be HTML.
    """
    return bool(re.search(r"</?[a-z][\s\S]*>", (text or "").strip(), re.IGNORECASE))
