# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ShopeeItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    name = scrapy.Field()
    avg_rating = scrapy.Field()
    n_reviews = scrapy.Field()
    n_sold = scrapy.Field()
    price = scrapy.Field()
    n_loved = scrapy.Field()
    rate_5 = scrapy.Field()
    rate_4 = scrapy.Field()
    rate_3 = scrapy.Field()
    rate_2 = scrapy.Field()
    rate_1 = scrapy.Field()
    rate_with_cmt = scrapy.Field()
    rate_with_imgvid = scrapy.Field()
    shop_name = scrapy.Field()
    shop_n_review = scrapy.Field()
    shop_n_product = scrapy.Field()
    shop_rate_feedback = scrapy.Field()
    shop_time_feedback = scrapy.Field()
    shop_age = scrapy.Field()
    shop_follower = scrapy.Field()

