from __future__ import annotations

import logging
import re


def answer_extract(reply: str, logger: logging.Logger) -> str:
    text = reply.strip()
    pattern_strict = r'answer[\"\'\s]*[:=][\"\'\s]*([A-Za-z])(?:\W|$)'
    match = re.search(pattern_strict, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    pattern_bracket = r'(?<!not\s)[\{\(\[\"\']\s*([A-Za-z])\s*[\}\)\]\"\']'
    matches = re.findall(pattern_bracket, text, re.IGNORECASE)
    if matches:
        if len(matches) == 1:
            return matches[0].upper()
        if len(text) < 50:
            return matches[-1].upper()

    if len(text) < 20 and "not" not in text.lower():
        match = re.search(r'([A-Za-z])', text)
        if match:
            return match.group(1).upper()

    logger.error("Failed to extract answer from: %s", reply)
    raise ValueError("No matching substring found")
