import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from langchain.schema import Document


def crawl_website(
    start_url: str,
    max_pages: int = 1,
    timeout: int = 5,
) -> list[Document]:
    """
    Crawl a website starting from a given URL and scrape content
    into LangChain Document objects.

    Args:
        start_url (str): The URL to start crawling from.
        max_pages (int, optional): Maximum number of pages to crawl.
        timeout (int, optional): Timeout for each request in seconds.

    Returns:
        list[Document]: A list of Document objects with page content + metadata.
    """
    visited = set()
    queue = deque([start_url])
    domain = urlparse(start_url).netloc
    documents: list[Document] = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        try:
            response = requests.get(
                url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}
            )
            if response.status_code != 200:
                continue

            visited.add(url)

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract visible text from important tags
            texts = []
            for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
                text = tag.get_text(strip=True)
                if text:
                    texts.append(text)

            page_content = "\n".join(texts).strip()
            if page_content:
                print(page_content)
                documents.append(
                    Document(page_content=page_content, metadata={"source": url})
                )

            # Enqueue same-domain links
            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == domain and full_url not in visited:
                    queue.append(full_url)

        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

    print(f"Crawled {len(visited)} pages → {len(documents)} documents created.")
    return documents


# crawl_website(start_url="https://react.dev/reference/react")
