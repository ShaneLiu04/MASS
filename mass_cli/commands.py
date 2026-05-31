"""
MASS CLI 命令行工具
提供诊断、回测、数据管理等命令
"""
import os
import sys
import json
import time
import click
from datetime import datetime, timedelta
from typing import List, Optional

from loguru import logger

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.orchestrator import AgentOrchestrator
from agent.models.database import Database
from agent.core.validator import DecisionValidator
from agent.core.cache import cache


@click.group()
@click.option('--mock', is_flag=True, help='使用Mock LLM模式')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
def cli(mock, verbose):
    """MASS 多智能体股票投研系统 CLI"""
    if mock:
        os.environ['USE_MOCK_LLM'] = 'True'
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")


@cli.command()
@click.argument('stock_code')
@click.option('--name', '-n', default='', help='股票名称')
@click.option('--json-output', '-j', is_flag=True, help='JSON格式输出')
def diagnose(stock_code, name, json_output):
    """对单只股票进行多智能体诊断"""
    stock_code = stock_code.strip().replace('.', '').replace('sh', '').replace('sz', '')
    
    if not stock_code.isdigit() or len(stock_code) != 6:
        click.echo(click.style("错误: 股票代码必须是6位数字", fg="red"))
        sys.exit(1)
    
    use_mock = os.getenv('USE_MOCK_LLM', 'False').lower() == 'true'
    orchestrator = AgentOrchestrator(use_mock_llm=use_mock)
    
    click.echo(f"正在诊断 {stock_code} {name or ''} ...")
    
    start = time.time()
    result = orchestrator.run_diagnosis(stock_code=stock_code, stock_name=name)
    elapsed = time.time() - start
    
    fd = result["final_decision"]
    decisions = {-1: ("卖出", "red"), 0: ("观望", "yellow"), 1: ("买入", "green")}
    decision_text, color = decisions.get(fd["decision"], ("未知", "white"))
    
    if json_output:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    
    click.echo(f"\n{'='*60}")
    click.echo(click.style(f"  {result['stock_name']} ({result['stock_code']})", bold=True))
    click.echo(f"  当前价格: ¥{result['current_price']:.2f}")
    click.echo(f"  市场周期: {result['market_cycle']}")
    click.echo(f"{'='*60}")
    
    click.echo(f"\nChairman 决策: ", nl=False)
    click.echo(click.style(decision_text, fg=color, bold=True))
    click.echo(f"  置信度:    {fd['confidence']*100:.1f}%")
    click.echo(f"  建议仓位:  {fd.get('position_pct', 0)*100:.1f}%")
    click.echo(f"  目标价:    ¥{fd.get('target_price', '--')}")
    click.echo(f"  止损价:    ¥{fd.get('stop_loss', '--')}")
    click.echo(f"  时间周期:  {fd.get('time_horizon', '--')}")
    click.echo(f"  预期收益:  {fd.get('expected_return_pct', 0):.1f}%")
    
    click.echo(f"\n{'-'*60}")
    click.echo("各Agent观点:")
    for agent_id, op in result["opinions"].items():
        sig_text, sig_color = decisions.get(op["signal"], ("?", "white"))
        click.echo(f"  {agent_id:12s}: ", nl=False)
        click.echo(click.style(f"{sig_text:4s}", fg=sig_color), nl=False)
        click.echo(f"  置信度 {op['confidence']*100:.0f}%")
    
    click.echo(f"\n{'-'*60}")
    click.echo(f"推理: {fd.get('reasoning', '--')}")
    
    consensus = fd.get("consensus_factors", [])
    if consensus:
        click.echo(f"\n共识因子:")
        for f in consensus:
            click.echo(f"  [OK] {f}")
    
    scenarios = fd.get("scenario_analysis", {})
    if scenarios:
        click.echo(f"\n情景分析:")
        for name, s in scenarios.items():
            click.echo(f"  {name}: 概率{s.get('probability',0)*100:.0f}%, 收益{s.get('return_pct',0):+.1f}%")
    
    click.echo(f"\n处理耗时: {elapsed:.2f}s")
    click.echo(f"\n{result.get('disclaimer', '')}")


@cli.command()
@click.argument('stock_codes', nargs=-1, required=True)
def batch(stock_codes):
    """批量诊断多只股票"""
    use_mock = os.getenv('USE_MOCK_LLM', 'False').lower() == 'true'
    orchestrator = AgentOrchestrator(use_mock_llm=use_mock)
    
    results = []
    for code in stock_codes:
        code = code.strip().replace('.', '')
        if not code.isdigit() or len(code) != 6:
            click.echo(click.style(f"跳过无效代码: {code}", fg="yellow"))
            continue
        
        click.echo(f"诊断 {code} ...", nl=False)
        try:
            result = orchestrator.run_diagnosis(stock_code=code)
            fd = result["final_decision"]
            decisions = {-1: "卖出", 0: "观望", 1: "买入"}
            sig = decisions.get(fd["decision"], "?")
            click.echo(click.style(f" {sig} (置信度{fd['confidence']*100:.0f}%)", fg="green" if fd['decision']==1 else ("red" if fd['decision']==-1 else "yellow")))
            results.append({"code": code, "decision": fd["decision"], "confidence": fd["confidence"]})
        except Exception as e:
            click.echo(click.style(f" 失败: {e}", fg="red"))
    
    click.echo(f"\n{'='*60}")
    click.echo(f"批量诊断完成: {len(results)}/{len(stock_codes)}")
    buy = sum(1 for r in results if r["decision"] == 1)
    sell = sum(1 for r in results if r["decision"] == -1)
    hold = sum(1 for r in results if r["decision"] == 0)
    click.echo(f"买入: {buy} | 观望: {hold} | 卖出: {sell}")


@cli.command()
@click.option('--days', default=30, help='验证天数范围')
def validate(days):
    """运行回测验证"""
    click.echo(f"运行最近 {days} 天的决策回测验证...")
    
    validator = DecisionValidator()
    report = validator.validate_yesterday_decisions()
    
    click.echo(f"\n{'='*60}")
    click.echo("回测报告")
    click.echo(f"{'='*60}")
    click.echo(f"验证决策数: {report['validated_count']}")
    click.echo(f"平均收益:   {report['avg_return']:+.2f}%")
    click.echo(f"达标率:     {report['hit_target_rate']:.1f}%")
    click.echo(f"止损率:     {report['hit_stop_rate']:.1f}%")
    
    if report['details']:
        click.echo(f"\n详情:")
        for d in report['details'][:10]:
            color = "green" if d['actual_return'] > 0 else "red"
            click.echo(click.style(f"  {d['stock_code']}: {d['actual_return']:+.2f}%", fg=color))


@cli.command()
def history():
    """查看历史决策记录"""
    db = Database()
    decisions = db.get_decisions(limit=20)
    
    if not decisions:
        click.echo("暂无历史记录")
        return
    
    click.echo(f"{'ID':>4} {'代码':>8} {'日期':>12} {'决策':>6} {'置信度':>8} {'仓位':>8} {'验证':>6}")
    click.echo("-" * 60)
    
    for d in decisions:
        sig_map = {-1: "卖出", 0: "观望", 1: "买入"}
        sig = sig_map.get(d["decision"], "?")
        validated = "Y" if d.get("validated") else "-"
        click.echo(f"{d['id']:>4} {d['stock_code']:>8} {d['decision_date']:>12} {sig:>6} {d['confidence']*100 if d['confidence'] else 0:>7.1f}% {d['position_pct']*100 if d['position_pct'] else 0:>7.1f}% {validated:>6}")


@cli.command()
def stats():
    """查看系统统计"""
    db = Database()
    db_stats = db.get_stats()
    cache_stats = cache.get_stats()
    
    click.echo(f"{'='*60}")
    click.echo("系统统计")
    click.echo(f"{'='*60}")
    click.echo(f"数据库:")
    click.echo(f"  总决策数:     {db_stats['total_decisions']}")
    click.echo(f"  模拟持仓:     {db_stats['total_positions']}")
    click.echo(f"  已验证:       {db_stats['validated_decisions']} ({db_stats['validation_rate']}%)")
    click.echo(f"缓存:")
    click.echo(f"  缓存大小:     {cache_stats['size']}")
    click.echo(f"  命中率:       {cache_stats['hit_rate']}%")
    click.echo(f"  命中/未命中:  {cache_stats['hits']}/{cache_stats['misses']}")


@cli.command()
def clear_cache():
    """清空系统缓存"""
    cache.clear()
    click.echo(click.style("缓存已清空", fg="green"))


@cli.command()
@click.argument('stock_code')
def position(stock_code):
    """查看/添加模拟持仓 (交互式)"""
    db = Database()
    positions = db.get_virtual_positions("default")
    
    # 查找是否已有持仓
    existing = [p for p in positions if p["stock_code"] == stock_code]
    
    if existing:
        click.echo(f"当前持仓 {stock_code}:")
        for p in existing:
            click.echo(f"  成本: ¥{p['entry_price']:.2f}, 数量: {p['shares']}, 仓位: {p['position_pct']*100:.1f}%")
    else:
        click.echo(f"无 {stock_code} 持仓")


if __name__ == '__main__':
    cli()
