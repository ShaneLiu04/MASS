"""
MASS 爬虫注册表
多源聚合与字段级合并 — 分层并发 + 优先级短路版
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from typing import Dict, Any, Optional, List, Set, Tuple

from loguru import logger

from agent.crawlers.base import BaseCrawler

# 模块级线程池：5 个爬虫源的并发 fetch
_CRAWLER_EXECUTOR = ThreadPoolExecutor(max_workers=5, thread_name_prefix="crawler-")
_CRAWLER_FETCH_TIMEOUT = 30  # 单个爬虫超时秒数

# 优先级分层间隔：同一批次内最高与最低优先级差距不超过此值
_PRIORITY_TIER_GAP = 15


class CrawlerRegistry:
    """
    多源爬虫注册表 — 字段级合并版（单例）

    全局唯一实例，所有 StockDataTool / SentimentTool 共享同一组爬虫和 Session。
    用法:
        registry = CrawlerRegistry.get_instance()
        data = registry.fetch_merge("600000", "fundamentals")
    """

    _instance: Optional["CrawlerRegistry"] = None
    _instance_lock = threading.Lock()

    # 关键字段：如果所有源都无法提供这些字段，视为获取失败
    CRITICAL_FIELDS = {
        "fundamentals": ["company_name"],
        "fund_flow": ["daily_flow"],
        "market_context": ["indices"],
        "macro": [],
        "sentiment": [],
        "news": [],
        "kline": ["df"],
    }

    def __init__(self):
        self._crawlers: List[BaseCrawler] = []
        self._crawler_names: Set[str] = set()
        self._failure_counts: Dict[str, int] = {}
        self._max_failures = 3

    @classmethod
    def get_instance(cls) -> "CrawlerRegistry":
        """获取全局单例"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(self, crawler: BaseCrawler) -> None:
        """注册爬虫源 — 同名自动跳过，幂等安全"""
        if not isinstance(crawler, BaseCrawler):
            raise TypeError("crawler 必须是 BaseCrawler 的子类")
        if crawler.name in self._crawler_names:
            return
        self._crawlers.append(crawler)
        self._crawler_names.add(crawler.name)
        self._crawlers.sort(key=lambda c: c.priority, reverse=True)
        logger.info(f"注册爬虫源: {crawler.name} (priority={crawler.priority})")

    def fetch(self, stock_code: str, data_type: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        传统单源获取：按优先级返回第一个成功的完整结果
        （保留向后兼容，但推荐用 fetch_merge）
        """
        for crawler in self._crawlers:
            name = crawler.name
            # 优先使用断路器状态判断
            cb = getattr(crawler, '_circuit_breaker', None)
            if cb is not None and not cb.can_execute():
                logger.debug(f"[{name}] 断路器 OPEN，跳过")
                continue
            # 保留原有失败计数作为后备
            if self._failure_counts.get(name, 0) >= self._max_failures:
                continue

            try:
                result = crawler.fetch(stock_code, data_type, **kwargs)
                if result is not None and isinstance(result, dict) and len(result) > 0:
                    self._failure_counts[name] = 0
                    return result
                self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
            except Exception as e:
                logger.warning(f"[{name}] 获取异常: {e}")
                self._failure_counts[name] = self._failure_counts.get(name, 0) + 1

        return None

    def fetch_merge(self, stock_code: str, data_type: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        字段级合并获取：分层并发 fan-out + 优先级短路

        按优先级将爬虫源分为若干批次，同批次内并发执行。
        每批次完成后检查关键字段是否已满足，若满足则提前返回，
        不再启动低优先级批次，减少无效请求和连接占用。

        返回结构:
        {
            "_meta": {
                "sources_tried": ["eastmoney", "sina", "tx"],
                "sources_succeeded": ["eastmoney", "sina"],
                "fields_coverage": {"pe_ttm": "eastmoney", "bid_ask": "sina"},
                "missing_fields": ["north_bound_5d"],
                "data_completeness": 0.75,
            },
            ...
        }
        """
        merged: Dict[str, Any] = {"_meta": {
            "sources_tried": [],
            "sources_succeeded": [],
            "fields_coverage": {},
            "missing_fields": [],
            "conflicts": [],
        }}
        all_fields_found: Set[str] = set()
        all_fields_from_sources: Dict[str, str] = {}
        # 记录每个字段的优先级，用于冲突检测
        field_priorities: Dict[str, int] = {}

        if not self._crawlers:
            logger.warning("没有注册任何爬虫源")
            return None

        # 预过滤：跳过断路器 OPEN 或失败次数过多的爬虫，避免浪费线程池资源
        active_crawlers = self._filter_active_crawlers(self._crawlers)
        if not active_crawlers:
            logger.warning("所有爬虫源均不可用（断路器开启或失败次数过多）")
            return None

        # 按优先级分批次（如 [EastMoney,Akshare,Sina], [Tx], [THS]）
        tiers = self._group_by_priority_tiers(active_crawlers)
        logger.debug(
            f"[fetch_merge] {stock_code}/{data_type} 分为 {len(tiers)} 个优先级批次: "
            f"{[[c.name for c in tier] for tier in tiers]}"
        )

        for tier_idx, tier_crawlers in enumerate(tiers):
            # 记录本批次尝试的源
            for crawler in tier_crawlers:
                merged["_meta"]["sources_tried"].append(crawler.name)

            # ── 提交本批次并发任务 ──
            futures_to_crawler: Dict[Any, BaseCrawler] = {}
            for crawler in tier_crawlers:
                future = _CRAWLER_EXECUTOR.submit(
                    self._safe_fetch, crawler, stock_code, data_type, kwargs
                )
                futures_to_crawler[future] = crawler

            # ── 收集本批次结果（先按完成顺序收，再按优先级排序合并）──
            tier_results: List[Tuple[int, str, Optional[Dict[str, Any]]]] = []
            for future in as_completed(futures_to_crawler):
                crawler = futures_to_crawler[future]
                try:
                    result = future.result(timeout=_CRAWLER_FETCH_TIMEOUT)
                    tier_results.append((crawler.priority, crawler.name, result))
                except FutureTimeoutError:
                    logger.warning(
                        f"[{crawler.name}] 并发 fetch 超时 ({_CRAWLER_FETCH_TIMEOUT}s)"
                    )
                    tier_results.append((crawler.priority, crawler.name, None))
                except Exception as e:
                    logger.warning(f"[{crawler.name}] 并发 fetch 异常: {e}")
                    tier_results.append((crawler.priority, crawler.name, None))

            # 按优先级降序排列后合并字段（高优先级源字段优先采用）
            tier_results.sort(key=lambda x: x[0], reverse=True)
            for priority, name, result in tier_results:
                self._merge_source_result(
                    merged, result, name, priority,
                    all_fields_found, all_fields_from_sources, field_priorities
                )

            # ── 优先级短路：关键字段已满足则提前返回 ──
            critical = self.CRITICAL_FIELDS.get(data_type, [])
            missing_critical = [f for f in critical if f not in all_fields_found]
            if not missing_critical:
                skipped = [c.name for tier in tiers[tier_idx + 1:] for c in tier]
                if skipped:
                    logger.info(
                        f"[fetch_merge] 关键字段已满足，短路返回，"
                        f"跳过低优先级源: {skipped}"
                    )
                break  # 不再启动后续低优先级批次

        # ── 最终检查 ──
        critical = self.CRITICAL_FIELDS.get(data_type, [])
        missing_critical = [f for f in critical if f not in all_fields_found]

        if missing_critical:
            logger.warning(
                f"关键字段缺失: {missing_critical}，所有源均无法获取 {data_type}"
            )
            return None

        if len(all_fields_found) == 0:
            logger.warning(f"所有爬虫源均无法获取 {data_type} for {stock_code}")
            return None

        merged["_meta"]["fields_coverage"] = all_fields_from_sources
        expected_fields = self._get_expected_fields(data_type)
        merged["_meta"]["missing_fields"] = [
            f for f in expected_fields if f not in all_fields_found
        ]
        total_expected = len(expected_fields) if expected_fields else len(all_fields_found)
        merged["_meta"]["data_completeness"] = (
            round(len(all_fields_found) / total_expected, 2) if total_expected > 0 else 1.0
        )

        return merged

    @staticmethod
    def _is_field_empty(value: Any) -> bool:
        """
        判断字段值是否真正为空。

        注意：0、0.0、False 等是合法值，不应视为空。
        """
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        # pandas DataFrame 判空
        if hasattr(value, "empty"):
            return bool(value.empty)
        return False

    # 冲突检测阈值：同一字段不同源差异超过此比例时视为异常
    _CONFLICT_THRESHOLD = 0.50

    def _merge_source_result(
        self,
        merged: Dict[str, Any],
        result: Optional[Dict[str, Any]],
        source_name: str,
        source_priority: int,
        all_fields_found: Set[str],
        all_fields_from_sources: Dict[str, str],
        field_priorities: Dict[str, int],
    ) -> None:
        """
        将单个爬虫源的结果合并到总结果中。

        特性:
        - 零容忍 mock 过滤
        - 字段级去重
        - 数值冲突检测：差异超过阈值时记录异常，低优先级源的异常值不覆盖
        """
        if result is None or not isinstance(result, dict):
            return

        # 零容忍：过滤 mock 数据
        if result.get("source") == "mock" or "mock" in str(result.get("source", "")):
            logger.warning(f"[{source_name}] 返回了mock数据，拒绝使用")
            return

        merged["_meta"]["sources_succeeded"].append(source_name)
        logger.info(f"[{source_name}] 成功获取数据 (priority={source_priority})")

        for key, value in result.items():
            if key.startswith("_"):
                continue
            if self._is_field_empty(value):
                continue

            if key in all_fields_found:
                # ── 冲突检测：同一字段不同源数值差异大 ──
                existing = merged[key]
                if (
                    isinstance(value, (int, float))
                    and isinstance(existing, (int, float))
                    and not isinstance(value, bool)
                    and not isinstance(existing, bool)
                ):
                    # 避免除零：以绝对值较大的为基准
                    base = max(abs(existing), abs(value), 1e-9)
                    diff_ratio = abs(value - existing) / base

                    if diff_ratio > self._CONFLICT_THRESHOLD:
                        conflict_info = {
                            "field": key,
                            "existing_source": all_fields_from_sources[key],
                            "existing_value": existing,
                            "incoming_source": source_name,
                            "incoming_value": value,
                            "diff_ratio": round(diff_ratio, 2),
                        }
                        merged["_meta"]["conflicts"].append(conflict_info)
                        logger.warning(
                            f"字段冲突 [{key}]: {all_fields_from_sources[key]}={existing}, "
                            f"{source_name}={value}, 差异 {diff_ratio * 100:.1f}%"
                        )

                        # 低优先级源的异常值，不覆盖高优先级的已有值
                        if source_priority < field_priorities.get(key, 0):
                            continue

                # 优先级比较：同优先级或更低不覆盖
                if source_priority <= field_priorities.get(key, 0):
                    continue

            merged[key] = value
            all_fields_found.add(key)
            all_fields_from_sources[key] = source_name
            field_priorities[key] = source_priority

    def _filter_active_crawlers(self, crawlers: List[BaseCrawler]) -> List[BaseCrawler]:
        """预过滤掉当前不可用的爬虫源（断路器 OPEN 或失败次数过多）"""
        active: List[BaseCrawler] = []
        for crawler in crawlers:
            # 断路器检查
            cb = getattr(crawler, "_circuit_breaker", None)
            if cb is not None and not cb.can_execute():
                logger.debug(f"[{crawler.name}] 断路器 OPEN，预过滤跳过")
                continue
            # 失败计数检查
            if self._failure_counts.get(crawler.name, 0) >= self._max_failures:
                logger.debug(f"[{crawler.name}] 失败次数过多，预过滤跳过")
                continue
            active.append(crawler)
        return active

    def _group_by_priority_tiers(self, crawlers: List[BaseCrawler]) -> List[List[BaseCrawler]]:
        """
        将爬虫按优先级分批次。

        策略：按优先级降序排列，相邻优先级差距超过 _PRIORITY_TIER_GAP 时开启新批次。
        这样高优先级源（如 EastMoney 100, Akshare 95, Sina 90）通常在同一批次并发，
        中优先级（Tx 80）在第二批，低优先级（THS 50）在第三批。
        """
        if not crawlers:
            return []

        sorted_crawlers = sorted(crawlers, key=lambda c: c.priority, reverse=True)
        tiers: List[List[BaseCrawler]] = []
        current_tier: List[BaseCrawler] = [sorted_crawlers[0]]

        for crawler in sorted_crawlers[1:]:
            tier_max_priority = current_tier[0].priority
            # 与当前批次最高优先级差距在阈值内则加入同批次
            if tier_max_priority - crawler.priority <= _PRIORITY_TIER_GAP:
                current_tier.append(crawler)
            else:
                tiers.append(current_tier)
                current_tier = [crawler]

        if current_tier:
            tiers.append(current_tier)

        return tiers

    @staticmethod
    def _safe_fetch(
        crawler: BaseCrawler,
        stock_code: str,
        data_type: str,
        kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """线程安全的单个爬虫 fetch 包装，异常不会传播到线程池外"""
        # 断路器快速失败：避免故障源占用线程池
        cb = getattr(crawler, '_circuit_breaker', None)
        if cb is not None and not cb.can_execute():
            logger.debug(f"[{crawler.name}] 断路器 OPEN，safe_fetch 快速失败")
            return None
        try:
            return crawler.fetch(stock_code, data_type, **kwargs)
        except Exception as e:
            logger.warning(f"[{crawler.name}] fetch 抛出异常: {e}")
            return None

    def _get_expected_fields(self, data_type: str) -> List[str]:
        """获取某数据类型期望的字段列表（用于计算覆盖率）"""
        fields_map = {
            "fundamentals": [
                "stock_code", "company_name", "industry", "latest_price",
                "pe_ttm", "pb", "roe", "market_cap", "total_revenue",
                "gross_margin", "net_margin", "revenue_yoy", "profit_yoy",
            ],
            "fund_flow": [
                "stock_code", "main_net_inflow_total", "main_inflow_days", "daily_flow",
            ],
            "market_context": [
                "indices", "sector_top",
            ],
            "sentiment": [
                "zt_count", "dt_count", "news_count_7d",
            ],
            "macro": [
                "bond_yield_10y", "pmi", "policy_stance",
            ],
            "news": [],
            "kline": ["df"],
        }
        return fields_map.get(data_type, [])

    def get_sources(self) -> List[str]:
        """获取所有已注册的爬虫源名称"""
        return [c.name for c in self._crawlers]

    def health_check(self) -> Dict[str, bool]:
        """健康检查所有爬虫源"""
        return {c.name: c.health_check() for c in self._crawlers}

    def reset_failures(self) -> None:
        """重置所有失败计数"""
        self._failure_counts.clear()
        logger.info("重置所有爬虫源失败计数")
