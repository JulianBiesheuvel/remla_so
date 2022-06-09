"""
Your friendly neighbourhood spider.
"""

# pylint: skip-file
# mypy: ignore-errors

import scrapy

from scraper.items import Question


class QuestionsSpider(scrapy.Spider):
    """A spider scraping StackOverflow question titles and tags."""

    name = "questions"
    start_urls = [
        "https://stackoverflow.com/questions?tab=newest&pagesize=50",
    ]

    def parse(self, response):
        """Scrapes the questions of a single page"""

        for question in response.css("#questions .s-post-summary"):
            yield Question(
                id=int(question.attrib["data-post-id"]),
                title=question.css("a.s-link::text").get(),
                tags=str(question.css(".tags .post-tag::text").getall()),
            )

        next_page = response.css(
            ".s-pagination.float-left a.s-pagination--item::attr(href)"
        ).getall()[-1]
        yield response.follow(next_page + "&pagesize=50", self.parse)
