"""
MASS 统一爬虫框架
支持多源数据获取、自动降级、反爬策略
"""
from agent.crawlers.base import BaseCrawler, DataQualityError
from agent.crawlers.registry import CrawlerRegistry
from agent.crawlers.eastmoney import EastMoneyCrawler
from agent.crawlers.ths import THSCrawler
from agent.crawlers.sina import SinaCrawler
from agent.crawlers.tx import TxCrawler
from agent.crawlers.akshare_crawler import AkshareCrawler

__all__ = [
    "BaseCrawler",
    "DataQualityError",
    "CrawlerRegistry",
    "EastMoneyCrawler",
    "THSCrawler",
    "SinaCrawler",
    "TxCrawler",
    "AkshareCrawler",
]
