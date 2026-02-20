# processing/cleaner.py
import re


def clean_text(text):
    """
    Clean raw extracted text:
    - Remove null bytes and control characters
    - Normalize Unicode whitespace
    - Collapse multiple spaces/newlines
    - Strip leading/trailing whitespace
    """
    # Remove null bytes and control characters (keep newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Replace tabs with spaces
    text = text.replace('\t', ' ')

    # Collapse multiple blank lines into a single newline
    text = re.sub(r'\n\s*\n', '\n', text)

    # Collapse multiple spaces into one
    text = re.sub(r' +', ' ', text)

    return text.strip()
