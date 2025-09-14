from rusenttokenize import ru_sent_tokenize
import re

def format_text(text: str) -> str:
    sentences = ru_sent_tokenize(text)

    processed = []
    for sent in sentences:
        # заменяем тире на "   -   " (с учётом разных вариантов тире)
        sent = re.sub(r"\s*[-—]\s*", "   —   ", sent)
        processed.append(sent.strip() + "   ")  # добавляем 2 пробела в конце

    return "".join(processed)
