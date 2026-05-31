#!/usr/bin/env python
"""
MASS 启动脚本
简化启动流程，支持命令行参数
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='MASS 多智能体股票投研系统')
    parser.add_argument('--mock', action='store_true', help='使用Mock LLM模式（无需API密钥）')
    parser.add_argument('--port', type=int, default=5000, help='服务端口 (默认5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址 (默认0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='开启Debug模式')
    parser.add_argument('--diagnose', type=str, help='命令行诊断股票 (如: --diagnose 000001)')
    
    args = parser.parse_args()
    
    # 设置环境变量
    if args.mock:
        os.environ['USE_MOCK_LLM'] = 'True'
        print('[INFO] 使用 Mock LLM 模式')
    
    os.environ['FLASK_PORT'] = str(args.port)
    os.environ['FLASK_HOST'] = args.host
    os.environ['FLASK_DEBUG'] = 'True' if args.debug else 'False'
    
    if args.diagnose:
        # 命令行单次诊断
        os.environ['USE_MOCK_LLM'] = 'True'
        from agent.core.orchestrator import AgentOrchestrator
        
        print(f'\n=== MASS 诊断: {args.diagnose} ===\n')
        orch = AgentOrchestrator(use_mock_llm=True)
        result = orch.run_diagnosis(stock_code=args.diagnose)
        
        fd = result['final_decision']
        decisions = {-1: '卖出', 0: '观望', 1: '买入'}
        
        print(f"股票: {result['stock_name']} ({result['stock_code']})")
        print(f"当前价: ¥{result['current_price']}")
        print(f"市场周期: {result['market_cycle']}")
        print(f"\n{'='*40}")
        print(f"Chairman 决策: {decisions.get(fd['decision'], '观望')}")
        print(f"置信度: {fd['confidence']*100:.1f}%")
        print(f"建议仓位: {fd.get('position_pct', 0)*100:.1f}%")
        print(f"目标价: ¥{fd.get('target_price', '--')}")
        print(f"止损价: ¥{fd.get('stop_loss', '--')}")
        print(f"时间周期: {fd.get('time_horizon', '--')}")
        print(f"{'='*40}\n")
        
        print("各Agent观点:")
        for agent_id, op in result['opinions'].items():
            sig = decisions.get(op['signal'], '观望')
            print(f"  {agent_id}: {sig} (置信度 {op['confidence']*100:.0f}%)")
        
        print(f"\n处理耗时: {result['processing_time_seconds']}s")
        print(f"\n{result.get('disclaimer', '')}")
        return
    
    # 启动Web服务
    from app import create_app
    app = create_app()
    
    print(f"\n{'='*50}")
    print(f"  MASS 多智能体股票投研系统 v1.0.0")
    print(f"{'='*50}")
    print(f"  服务地址: http://{args.host}:{args.port}")
    print(f"  Mock模式: {os.getenv('USE_MOCK_LLM', 'False')}")
    print(f"  Debug模式: {os.getenv('FLASK_DEBUG', 'False')}")
    print(f"{'='*50}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
