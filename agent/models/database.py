"""
MASS 数据库层
SQLite 封装：决策记录、Agent观点、模拟持仓
线程安全：每个线程拥有独立的数据库连接
"""
import sqlite3
import json
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from loguru import logger

from config import DATABASE_PATH


class _ConnectionPool:
    """线程本地 SQLite 连接池 — 避免每次查询新建连接"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
    
    def get(self) -> sqlite3.Connection:
        """获取当前线程的连接，不存在则创建"""
        conn = getattr(self._local, 'connection', None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # 启用 WAL 模式提升并发性能
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.connection = conn
            logger.debug(f"创建线程连接: {threading.current_thread().name}")
        return conn
    
    def close(self) -> None:
        """关闭当前线程的连接"""
        conn = getattr(self._local, 'connection', None)
        if conn is not None:
            conn.close()
            self._local.connection = None


class Database:
    """SQLite 数据库管理器 — 线程安全连接池"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(DATABASE_PATH)
        self._pool = _ConnectionPool(self.db_path)
        self._init_db()
    
    def _get_connection(self):
        return self._pool.get()
    
    def _init_db(self):
        """初始化数据库表结构"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        
        # Agent 决策记录表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code CHAR(6) NOT NULL,
                stock_name CHAR(64),
                decision_date CHAR(10) NOT NULL,
                decision_time CHAR(8) NOT NULL,
                decision INTEGER NOT NULL,
                confidence REAL,
                position_pct REAL,
                target_price REAL,
                stop_loss REAL,
                expected_return_pct REAL,
                reasoning TEXT,
                raw_json TEXT,
                actual_return_pct REAL,
                hit_target INTEGER,
                hit_stop_loss INTEGER,
                validated INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Agent 观点明细表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_opinions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id INTEGER,
                agent_id CHAR(16) NOT NULL,
                signal INTEGER NOT NULL,
                confidence REAL,
                reasoning TEXT,
                raw_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (decision_id) REFERENCES agent_decisions(id)
            )
        """)
        
        # 用户模拟持仓表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS virtual_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username CHAR(64),
                stock_code CHAR(6),
                stock_name CHAR(64),
                entry_price REAL,
                shares INTEGER,
                position_pct REAL,
                target_price REAL,
                stop_loss REAL,
                entry_date CHAR(10),
                status INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 回测验证记录表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS validation_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id INTEGER,
                validate_date CHAR(10),
                actual_return_pct REAL,
                hit_target INTEGER,
                hit_stop_loss INTEGER,
                close_price REAL,
                notes TEXT,
                FOREIGN KEY (decision_id) REFERENCES agent_decisions(id)
            )
        """)
        
        # Agent 准确率统计表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_accuracy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id CHAR(16) NOT NULL,
                month CHAR(7) NOT NULL,
                total_signals INTEGER DEFAULT 0,
                correct_signals INTEGER DEFAULT 0,
                accuracy REAL,
                avg_confidence REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # v2.3: 预测记录与精度追踪
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code CHAR(6) NOT NULL,
                stock_name CHAR(64),
                horizon CHAR(8) NOT NULL,
                risk_tolerance CHAR(16) DEFAULT 'moderate',
                investment_style CHAR(16) DEFAULT 'swing',
                predicted_direction CHAR(8) NOT NULL,
                raw_confidence REAL,
                calibrated_confidence REAL,
                data_quality_factor REAL,
                target_price_low REAL,
                target_price_high REAL,
                stop_loss REAL,
                probability_up REAL,
                probability_down REAL,
                probability_sideways REAL,
                holding_period_days INTEGER,
                model_used CHAR(64),
                fallback_used INTEGER DEFAULT 0,
                prompt_tokens_estimated INTEGER,
                reasoning TEXT,
                key_drivers TEXT,
                risk_factors TEXT,
                prediction_time CHAR(19) NOT NULL,
                entry_price REAL,
                -- 验证字段（N 日后回填）
                validated INTEGER DEFAULT 0,
                validate_date CHAR(10),
                actual_close_price REAL,
                actual_return_pct REAL,
                direction_correct INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # v2.4: 系统设置表（支持 Web UI 配置 API Key 等）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                config_key TEXT PRIMARY KEY,
                config_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("数据库初始化完成")
    
    def save_decision(self, package: Dict[str, Any]) -> int:
        """保存完整决策包，返回decision_id"""
        decision = package.get("final_decision", {})
        stock_code = package.get("stock_code", "")
        stock_name = package.get("stock_name", "")
        
        # 数据校验：防止虚拟/mock数据污染数据库
        if not stock_name or stock_name == stock_code or "模拟" in stock_name or "mock" in stock_name.lower():
            raw = package
            fundamentals = raw.get("fundamentals", {})
            real_name = fundamentals.get("company_name", "")
            if real_name and "模拟" not in real_name and "mock" not in real_name.lower():
                stock_name = real_name
            else:
                stock_name = stock_code
        
        conn = self._get_connection()
        cursor = conn.execute(
            """
            INSERT INTO agent_decisions
            (stock_code, stock_name, decision_date, decision_time, decision,
             confidence, position_pct, target_price, stop_loss,
             expected_return_pct, reasoning, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stock_code,
                stock_name,
                package.get("decision_date", ""),
                package.get("decision_time", ""),
                decision.get("decision", 0),
                decision.get("confidence", 0),
                decision.get("position_pct", 0),
                decision.get("target_price"),
                decision.get("stop_loss"),
                decision.get("expected_return_pct", 0),
                decision.get("reasoning", ""),
                json.dumps(package, ensure_ascii=False, default=str),
            ),
        )
        decision_id = cursor.lastrowid
        
        # 保存各Agent观点
        opinions = package.get("opinions", {})
        for agent_id, op in opinions.items():
            conn.execute(
                """
                INSERT INTO agent_opinions
                (decision_id, agent_id, signal, confidence, reasoning, raw_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    agent_id,
                    op.get("signal", 0),
                    op.get("confidence", 0),
                    op.get("reasoning", ""),
                    json.dumps(op, ensure_ascii=False, default=str),
                ),
            )
        
        conn.commit()
        logger.info(f"决策已保存: decision_id={decision_id}, stock={package.get('stock_code')}")
        return decision_id
    
    def get_decisions(
        self,
        stock_code: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """获取历史决策记录"""
        conn = self._get_connection()
        if stock_code:
            rows = conn.execute(
                "SELECT * FROM agent_decisions WHERE stock_code = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (stock_code, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM agent_decisions ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        
        return [dict(row) for row in rows]
    
    def get_decision_by_id(self, decision_id: int) -> Optional[Dict[str, Any]]:
        """获取单条决策详情"""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM agent_decisions WHERE id = ?",
            (decision_id,),
        ).fetchone()
        if row:
            result = dict(row)
            opinions = conn.execute(
                "SELECT * FROM agent_opinions WHERE decision_id = ?",
                (decision_id,),
            ).fetchall()
            result["opinions"] = [dict(o) for o in opinions]
            return result
        return None
    
    def add_virtual_position(self, position: Dict[str, Any]) -> int:
        """添加模拟持仓"""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            INSERT INTO virtual_positions
            (username, stock_code, stock_name, entry_price, shares,
             position_pct, target_price, stop_loss, entry_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.get("username", "default"),
                position.get("stock_code", ""),
                position.get("stock_name", ""),
                position.get("entry_price", 0),
                position.get("shares", 0),
                position.get("position_pct", 0),
                position.get("target_price"),
                position.get("stop_loss"),
                position.get("entry_date", datetime.now().strftime("%Y-%m-%d")),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    
    def get_virtual_positions(self, username: str = "default") -> List[Dict[str, Any]]:
        """获取用户模拟持仓"""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM virtual_positions WHERE username = ? AND status = 1 ORDER BY created_at DESC",
            (username,),
        ).fetchall()
        return [dict(row) for row in rows]
    
    def close_virtual_position(self, position_id: int, exit_price: float) -> bool:
        """平仓模拟持仓"""
        conn = self._get_connection()
        conn.execute(
            "UPDATE virtual_positions SET status = 0 WHERE id = ?",
            (position_id,),
        )
        conn.commit()
        return True
    
    def update_validation(self, decision_id: int, validation: Dict[str, Any]) -> bool:
        """更新回测验证结果"""
        conn = self._get_connection()
        conn.execute(
            """
            UPDATE agent_decisions
            SET actual_return_pct = ?, hit_target = ?, hit_stop_loss = ?, validated = 1
            WHERE id = ?
            """,
            (
                validation.get("actual_return_pct", 0),
                validation.get("hit_target", 0),
                validation.get("hit_stop_loss", 0),
                decision_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO validation_records
            (decision_id, validate_date, actual_return_pct, hit_target, hit_stop_loss, close_price, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                datetime.now().strftime("%Y-%m-%d"),
                validation.get("actual_return_pct", 0),
                validation.get("hit_target", 0),
                validation.get("hit_stop_loss", 0),
                validation.get("close_price", 0),
                validation.get("notes", ""),
            ),
        )
        conn.commit()
        return True
    
    def get_unvalidated_decisions(self, days: int = 30) -> List[Dict[str, Any]]:
        """获取未验证的决策（用于每日回测）"""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT * FROM agent_decisions
            WHERE validated = 0
            AND decision_date >= date('now', '-{} days')
            ORDER BY decision_date DESC
            """.format(days),
        ).fetchall()
        return [dict(row) for row in rows]
    
    def get_agent_accuracy_stats(self, month: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取Agent准确率统计"""
        conn = self._get_connection()
        if month:
            rows = conn.execute(
                "SELECT * FROM agent_accuracy WHERE month = ? ORDER BY accuracy DESC",
                (month,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM agent_accuracy ORDER BY month DESC, accuracy DESC",
            ).fetchall()
        return [dict(row) for row in rows]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计"""
        conn = self._get_connection()
        total_decisions = conn.execute(
            "SELECT COUNT(*) FROM agent_decisions"
        ).fetchone()[0]
        total_positions = conn.execute(
            "SELECT COUNT(*) FROM virtual_positions"
        ).fetchone()[0]
        validated = conn.execute(
            "SELECT COUNT(*) FROM agent_decisions WHERE validated = 1"
        ).fetchone()[0]
        
        return {
            "total_decisions": total_decisions,
            "total_positions": total_positions,
            "validated_decisions": validated,
            "validation_rate": round(validated / total_decisions * 100, 1) if total_decisions > 0 else 0,
        }

    # ══════════════════════════════════════════════════════════════════════
    # v2.3: 预测记录与精度追踪
    # ══════════════════════════════════════════════════════════════════════

    def save_prediction(self, result: "PredictionResult") -> int:
        """保存预测记录"""
        import json
        conn = self._get_connection()
        cursor = conn.execute(
            """
            INSERT INTO prediction_records
            (stock_code, stock_name, horizon, risk_tolerance, investment_style,
             predicted_direction, raw_confidence, calibrated_confidence,
             data_quality_factor, target_price_low, target_price_high, stop_loss,
             probability_up, probability_down, probability_sideways,
             holding_period_days, model_used, fallback_used, prompt_tokens_estimated,
             reasoning, key_drivers, risk_factors, prediction_time, entry_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.stock_code,
                result.stock_name,
                result.prediction_horizon,
                result.risk_tolerance,
                result.investment_style,
                result.direction,
                result.confidence,
                result.confidence_calibrated,
                result.data_quality_factor,
                result.target_price_low,
                result.target_price_high,
                result.stop_loss,
                result.probability_up,
                result.probability_down,
                result.probability_sideways,
                result.holding_period_days,
                result.model_used,
                1 if result.fallback_used else 0,
                result.prompt_tokens_estimated,
                result.reasoning,
                json.dumps(result.key_drivers, ensure_ascii=False),
                json.dumps(result.risk_factors, ensure_ascii=False),
                result.prediction_time,
                result.target_price_low if result.direction == "上涨" else (
                    result.target_price_high if result.direction == "下跌" else None
                ),
            ),
        )
        conn.commit()
        logger.info(f"预测记录已保存: id={cursor.lastrowid}, {result.stock_code}/{result.prediction_horizon}")
        return cursor.lastrowid

    def get_prediction_history(
        self,
        stock_code: Optional[str] = None,
        horizon: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """获取预测历史"""
        conn = self._get_connection()
        q = "SELECT * FROM prediction_records WHERE 1=1"
        params = []
        if stock_code:
            q += " AND stock_code = ?"
            params.append(stock_code)
        if horizon:
            q += " AND horizon = ?"
            params.append(horizon)
        q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return [dict(row) for row in conn.execute(q, params).fetchall()]

    def get_prediction_accuracy_stats(
        self, days: int = 30
    ) -> Dict[str, Any]:
        """获取预测准确率统计（仅统计已验证的）"""
        conn = self._get_connection()
        total = conn.execute(
            "SELECT COUNT(*) FROM prediction_records WHERE validated = 1"
        ).fetchone()[0]
        correct = conn.execute(
            "SELECT COUNT(*) FROM prediction_records WHERE validated = 1 AND direction_correct = 1"
        ).fetchone()[0]

        # 按 horizon 分组
        by_horizon = {}
        for row in conn.execute(
            """SELECT horizon, COUNT(*) as total,
                      SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as correct
               FROM prediction_records
               WHERE validated = 1
               GROUP BY horizon"""
        ).fetchall():
            by_horizon[row["horizon"]] = {
                "total": row["total"],
                "correct": row["correct"],
                "accuracy": round(row["correct"] / row["total"] * 100, 1) if row["total"] > 0 else 0,
            }

        return {
            "total_validated_predictions": total,
            "correct_predictions": correct,
            "overall_accuracy": round(correct / total * 100, 1) if total > 0 else 0,
            "by_horizon": by_horizon,
        }
