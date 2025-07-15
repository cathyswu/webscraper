import requests
url = "https://docs.pytorch.org/docs/stable/amp.html"  # Replace with the actual URL
response = requests.get(url)
html_content = response.text

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

all_text = soup.get_text()
print(all_text)

# Extract text from the first <h1> tag
title_element = soup.find('h1')
if title_element:
    title_text = title_element.get_text()
    print(f"Title: {title_text}")

# Extract text from all <p> tags
paragraphs = soup.find_all('p')
for p in paragraphs:
    paragraph_text = p.get_text()
    print(f"Paragraph: {paragraph_text}")