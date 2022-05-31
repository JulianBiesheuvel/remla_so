import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = [
        "https://stackoverflow.com/questions?tab=newest&pagesize=50",
        "https://stackoverflow.com/questions?tab=newest&page=2",
    ]

    def parse(self, response):
        for question in response.css("#questions .s-post-summary"):
            item = {
                "title": question.css("a.s-link::text").get(),
                "tags": question.css(".tags .post-tag::text").getall(),
            }
            yield item

        next_page = response.css(
            ".s-pagination.float-left a.s-pagination--item::attr(href)"
        ).getall()[-1]
        yield response.follow(next_page + "&pagesize=50", self.parse)
