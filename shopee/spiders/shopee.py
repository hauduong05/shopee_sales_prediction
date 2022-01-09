import scrapy
from scrapy_splash import SplashRequest
from shopee.items import ShopeeItem


class ShopeeSpider(scrapy.Spider):
    name = 'shopee'
    allowed_domains = ['shopee.vn']
    start_urls = ["https://shopee.vn/Ph%E1%BB%A5-ki%E1%BB%87n-tivi-cat.11036132.11036167",
                  # 'https://shopee.vn/%C4%90i%E1%BB%87n-tho%E1%BA%A1i-cat.11036030.11036031',
                  # 'https://shopee.vn/M%C3%A1y-t%C3%ADnh-b%E1%BA%A3ng-cat.11036030.11036041',
                  # 'https://shopee.vn/Pin-D%E1%BB%B1-Ph%C3%B2ng-cat.11036030.11036048',
                  # 'https://shopee.vn/Pin-G%E1%BA%AFn-Trong-C%C3%A1p-v%C3%A0-B%E1%BB%99-S%E1%BA%A1c-cat.11036030.11036054',
                  # 'https://shopee.vn/%E1%BB%90p-l%C6%B0ng-bao-da-Mi%E1%BA%BFng-d%C3%A1n-%C4%91i%E1%BB%87n-tho%E1%BA%A1i-cat.11036030.11036060',
                  # 'https://shopee.vn/B%E1%BA%A3o-v%E1%BB%87-m%C3%A0n-h%C3%ACnh-cat.11036030.11036064',
                  # 'https://shopee.vn/%C4%90%E1%BA%BF-gi%E1%BB%AF-%C4%91i%E1%BB%87n-tho%E1%BA%A1i-G%E1%BA%ADy-Ch%E1%BB%A5p-h%C3%ACnh-cat.11036030.11036074',
                  # 'https://shopee.vn/Th%E1%BA%BB-nh%E1%BB%9B-cat.11036030.11036083',
                  # 'https://shopee.vn/%C4%90i%E1%BB%87n-Tho%E1%BA%A1i-B%C3%A0n-cat.11036030.11036097',
                  # 'https://shopee.vn/M%C3%A1y-T%C3%ADnh-B%C3%A0n-cat.11035954.11035955',
                  # 'https://shopee.vn/M%C3%A0n-H%C3%ACnh-cat.11035954.11035961',
                  # 'https://shopee.vn/Linh-Ki%E1%BB%87n-M%C3%A1y-T%C3%ADnh-cat.11035954.11035962',
                  # 'https://shopee.vn/Thi%E1%BA%BFt-B%E1%BB%8B-M%E1%BA%A1ng-cat.11035954.11035983',
                  # 'https://shopee.vn/Thi%E1%BA%BFt-b%E1%BB%8B-%C4%91eo-th%C3%B4ng-minh-cat.11036132.11036160'
                  ]

    render_script = """
        function main(splash)
            local url = splash.args.url
            assert(splash:go(url))
            assert(splash:wait(5))

            return {
                html = splash:html(),
                url = splash:url(),
            }
        end
        """

    def start_requests(self):
        for url in self.start_urls:
            yield SplashRequest(
                url,
                self.parse_pages,
                endpoint='render.html',
                args={
                    'wait': 5,
                    'lua_source': self.render_script,
                }
            )

    def parse_pages(self, response, **kwargs):
        n_pages = int(response.css("span.shopee-mini-page-controller__total::text").extract_first())
        print(n_pages)
        for i in range(n_pages):
            yield SplashRequest(
                url=response.url + f'?page={i}&sortBy=pop',
                callback=self.parse_urls,
                endpoint='render.html',
                args={
                    'wait': 5,
                    'lua_source': self.render_script
                }
            )

    def parse_urls(self, response):
        urls = response.xpath('//div[contains(@class, "shopee-search-item-result__item")]/a/@href').getall()
        for url in urls:
            yield SplashRequest(
                url=response.urljoin(url),
                callback=self.parse_detail,
                endpoint='render.html',
                args={
                    'wait': 5,
                    'lua_source': self.render_script
                }
            )

    def parse_detail(self, response, **kwargs):
        item = ShopeeItem()

        item["name"] = response.css("div._3g8My- span::text").extract_first()
        item['avg_rating'] = response.css("div._3uBhVI.URjL1D::text").extract_first()
        item['n_sold'] = response.css("div._3b2Btx::text").extract_first()
        item["price"] = response.css("div._2v0Hgx::text").extract_first()
        item['n_loved'] = response.css('div.flex.items-center._3oXgUo div.Rs4O3p::text').extract_first()
        n_ratings = response.css('div.product-rating-overview__filter::text').extract()
        if n_ratings:

            item['rate_5'] = n_ratings[1]
            item['rate_4'] = n_ratings[2]
            item['rate_3'] = n_ratings[3]
            item['rate_2'] = n_ratings[4]
            item['rate_1'] = n_ratings[5]
            item['rate_with_cmt'] = n_ratings[6]
            item['rate_with_imgvid'] = n_ratings[7]
        item['shop_name'] = response.css('div._1wVLAc::text').extract_first()
        shop_info = response.css('span._33OqNH::text').extract()
        if shop_info:
            item['shop_n_review'] = shop_info[0]
            item['shop_n_product'] = shop_info[1]
            item['shop_rate_feedback'] = shop_info[2]
            item['shop_time_feedback'] = shop_info[3]
            item['shop_age'] = shop_info[4]
            item['shop_follower'] = shop_info[5]

        yield item
