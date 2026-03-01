import scrapy

class youm7scrap(scrapy.Spider):
    name = 'youm7scrap'
    start_urls = ['https://www.youm7.com/',
                  'https://www.youm7.com/Section/%D8%A7%D9%82%D8%AA%D8%B5%D8%A7%D8%AF-%D9%88%D8%A8%D9%88%D8%B1%D8%B5%D8%A9/297/1',
                  'https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%A7%D9%84%D8%B1%D9%8A%D8%A7%D8%B6%D8%A9/298/1',
                  'https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1']
    allowed_domains = ['youm7.com']

    def parse(self, response):
        paragraphs = response.xpath('//p//text()').getall()
        headings = response.xpath('//h1//text()').getall() + \
                   response.xpath('//h2//text()').getall() + \
                   response.xpath('//h3//text()').getall() + \
                   response.xpath('//h4//text()').getall() + \
                   response.xpath('//h5//text()').getall() + \
                   response.xpath('//h6//text()').getall()

        text_data = paragraphs + headings

        clean_text = ' '.join(text_data)

        yield {
            'url': response.url,
            'text': clean_text
        }

        for next_page in response.xpath('//a/@href').getall():
            yield response.follow(next_page, self.parse)