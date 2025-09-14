from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document
from urllib.parse import urljoin, urlparse


def crawl_js_website(start_url: str, max_pages: int = 5) -> list[Document]:
    """
    Crawl a JavaScript-rendered website (SPA) using WebBaseLoader,
    and return LangChain Document objects for the parent page and nested URLs.

    Args:
        start_url (str): The URL to start crawling from.
        max_pages (int, optional): Maximum number of pages to crawl.

    Returns:
        list[Document]: List of LangChain Documents containing page content.
    """
    visited = set()
    queue = [start_url]
    documents: list[Document] = []

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue

        try:
            loader = WebBaseLoader(url)
            docs: list[Document] = loader.load()
            if docs:
                documents.extend(docs)

            visited.add(url)

            # Find internal links from the URL structure (SPA hash links)
            parsed_url = urlparse(url)
            # Example: convert "#/quickstart" to full URL
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            if "#" in url:
                base = url.split("#")[0]
            else:
                base = url

            for doc in docs:
                # Extract hash-based links from the page content (simple approach)
                for line in doc.page_content.splitlines():
                    if line.startswith("#/") or line.startswith("/#"):
                        full_url = urljoin(base_url, line.strip())
                        if full_url not in visited and full_url not in queue:
                            queue.append(full_url)

        except Exception as e:
            print(f"Failed to load {url}: {e}")

    print(f"Crawled {len(visited)} pages → {len(documents)} documents created.")
    return documents


# Example usage:
# docs = crawl_js_website("https://docsify.js.org/#", max_pages=10)
# print(docs[0].page_content[:300])  # preview first doc
