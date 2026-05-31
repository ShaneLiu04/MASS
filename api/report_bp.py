"""
MASS Flask Blueprint: report_bp
"""
from flask import Blueprint, jsonify, request, current_app
from loguru import logger

from api.common import (
    get_orchestrator, get_db,
    safe_save_decision, safe_save_prediction,
    generate_cache_key, aggregate_portfolio_risk,
    cache, PORTFOLIO_EXECUTOR, DIAGNOSIS_EXECUTOR, DB_SAVE_EXECUTOR,
)
from api.middleware import RateLimiter

report_bp = Blueprint('report', __name__, url_prefix='/api/agent')

@report_bp.route('/report/generate', methods=['POST'])
@RateLimiter().limit(max_requests=5, window=60)
def generate_report():
    """
    生成投研报告
    
    Request:
        {"stock_code": "000001", "format": "json"}
    """
    try:
        data = request.get_json(force=True) or {}
        stock_code = data.get('stock_code', '').strip()
        fmt = data.get('format', 'json')
        
        if not stock_code:
            return jsonify({"error": "stock_code 不能为空"}), 400
        
        orchestrator = _get_orchestrator()
        result = orchestrator.run_diagnosis(stock_code=stock_code)
        
        if fmt == 'markdown':
            fd = result.get('final_decision', {})
            ops = result.get('opinions', {})
            sig_text = {1: '买入', 0: '观望', -1: '卖出'}.get(fd.get('decision', 0), '观望')
            
            md = f"# MASS投研报告: {result.get('stock_name', stock_code)}\n\n"
            md += f"**生成时间**: {result.get('decision_date')} {result.get('decision_time')}\n\n"
            md += f"## 综合决策\n\n"
            md += f"- **决策**: {sig_text} (置信度: {fd.get('confidence', 0)*100:.1f}%)\n"
            md += f"- **建议仓位**: {fd.get('position_pct', 0)*100:.1f}%\n"
            md += f"- **目标价**: ¥{fd.get('target_price', '--')}\n"
            md += f"- **止损价**: ¥{fd.get('stop_loss', '--')}\n"
            md += f"- **预期收益**: {fd.get('expected_return_pct', 0)}%\n\n"
            md += f"## 各Agent观点\n\n"
            for agent_id, op in ops.items():
                s = {1: '买入', 0: '观望', -1: '卖出'}.get(op.get('signal', 0), '观望')
                md += f"### {agent_id}\n- 信号: {s} (置信度: {op.get('confidence', 0)*100:.0f}%)\n"
                md += f"- 理由: {op.get('reasoning', '')}\n\n"
            md += f"## Chairman推理\n\n{fd.get('reasoning', '')}\n\n"
            md += "---\n*免责声明: 本报告仅供参考，不构成投资建议。*"
            
            return jsonify({"format": "markdown", "content": md})
        
        return jsonify({"format": "json", "data": result})
    
    except Exception as e:
        logger.exception("研报生成接口异常")
        return jsonify({"error": str(e)}), 500


