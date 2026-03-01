import re
import json
import spacy
from parsel import Selector
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

nlp = spacy.blank('ar')
nlp.add_pipe('sentencizer')

class MockResponse:
    def __init__(self, url, html_content):
        self.url = url
        self.selector = Selector(text=html_content)

    def xpath(self, query):
        return self.selector.xpath(query)

    def urljoin(self, link):
        return urljoin(self.url, link)
    
def remove_arabic_diacritics(text):
    text = re.sub(r'[\u0610-\u061A\u064B-\u0652\u0670\u0640]', '', text)
    return text

def clean_text(text):
    clean_text = BeautifulSoup(text, "html.parser").get_text()
    clean_text = remove_arabic_diacritics(clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = re.sub(r'[^\w\s.,?!]','', clean_text)
    clean_text = re.sub(r'[٠-٩]', lambda x: str(ord(x.group(0)) - 1632), clean_text)
    return clean_text.strip()

def preprocess_arabic_text(text):
    cleaned_text = clean_text(text)

    doc = nlp(cleaned_text)

    lemmatized_tokens = [token.lemma_ if token.lemma_ else token.text for token in doc]

    return ' '.join(lemmatized_tokens)

def chunk_text(text, max_chunk_size=1000):
    doc = nlp(text)
    
    chunks = []
    current_chunk = ""
    
    for sent in doc.sents:
        sentence = sent.text
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def extract_metadata(response):
    title = response.xpath('//title/text()').get()
    
    headers = []
    for i in range(1, 7):
        extracted = response.xpath(f'//h{i}//text()').getall()
        if extracted:
            headers.extend(extracted)

    links = response.xpath('//a/@href').getall()
    internal_links = []
    external_links = []
    
    for link in links:
        absolute_url = response.urljoin(link)
        netloc = urlparse(absolute_url).netloc
        
        if netloc == urlparse(response.url).netloc:
            internal_links.append(absolute_url)
        else:
            external_links.append(absolute_url)
    
    return {
        'title': title,
        'headers': headers,
        'internal_links': internal_links,
        'external_links': external_links
    }

def process_scraped_data(response, output_file='preprocessing/output/processed_data.json'):
    raw_text = response.xpath('//body//text()').getall()
    
    preprocessed_text = preprocess_arabic_text(' '.join(raw_text))
    print(f"Preprocessed text for URL {response.url}: {preprocessed_text[:200]}...")
    
    text_chunks = chunk_text(preprocessed_text)
    print(f"Text chunks for URL {response.url}: {text_chunks[:2]}...")
    
    metadata = extract_metadata(response)
    
    processed_headers = [preprocess_arabic_text(header) for header in metadata['headers'] if header.strip()]
    
    processed_title = preprocess_arabic_text(metadata['title']) if metadata['title'] else ""
    
    result = {
        'url': response.url,
        'title': processed_title,
        'headers': processed_headers,
        'internal_links': metadata['internal_links'],
        'external_links': metadata['external_links'],
        'text_chunks': text_chunks
    }

    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False) 
        f.write('\n')

    return result

def process_scraped_json(json_file, output_file='preprocessing/output/processed_data.json'):
    with open(json_file, 'r', encoding='utf-8') as f:
        scraped_data = json.load(f) 
        
        for data in scraped_data:
            url = data.get('url')
            html_content = data.get('text')

            if url and html_content:
                response = MockResponse(url, html_content)
                
                processed_data = process_scraped_data(response, output_file=output_file)
                print(f"Successfully processed: {url}")

if __name__ == "__main__":
    process_scraped_json('1_data_collection/datacollection1/output/scrap.json', output_file='2_preprocessing/output/processed_youm7_data.json')