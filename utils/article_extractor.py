import trafilatura

def extract_article(url):

try:

    downloaded = trafilatura.fetch_url(
        url
    )

    if not downloaded:

        return ""

    text = trafilatura.extract(
        downloaded
    )

    return text if text else ""

except Exception:

    return ""
