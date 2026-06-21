import re
import trafilatura


def extract_article(url):

    try:

        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return ""

        text = trafilatura.extract(
            downloaded
        )

        if not text:
            return ""

        text = re.sub(
            r'\s+',
            ' ',
            text
        ).strip()

        return text

    except Exception:

        return ""
