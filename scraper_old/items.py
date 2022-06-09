"""
Model for scraped data.
"""
import scrapy

class Question(scrapy.Item):
    """StackOverflow question representation"""
    id = scrapy.Field()
    title = scrapy.Field()
    tags = scrapy.Field()
