from bs4 import BeautifulSoup


def extract_code_snippets_from_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    return [code.get_text() for code in soup.find_all('code')]