"""
MASS 生产部署脚本
使用 waitress WSGI 服务器运行 Flask 应用

用法:
    python deploy.py          # 前台运行
    python deploy.py --daemon # 后台运行 (Windows: 使用 start 命令)
    python deploy.py --stop   # 停止服务
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

from loguru import logger

# 项目根目录
BASE_DIR = Path(__file__).parent.resolve()
os.chdir(BASE_DIR)


def check_environment():
    """检查部署环境"""
    checks = []

    # 1. .env 文件
    env_file = BASE_DIR / ".env"
    checks.append((".env 配置文件", env_file.exists()))

    # 2. 数据库
    db_file = BASE_DIR / "data" / "mass.db"
    checks.append(("SQLite 数据库", db_file.exists()))

    # 3. 日志目录
    log_dir = BASE_DIR / "logs"
    checks.append(("日志目录", log_dir.exists()))

    # 4. 静态资源
    css_file = BASE_DIR / "static" / "css" / "mass-theme.css"
    checks.append(("CSS 主题文件", css_file.exists()))

    # 5. 模板
    base_template = BASE_DIR / "templates" / "base.html"
    checks.append(("基础模板", base_template.exists()))

    # 6. waitress
    try:
        import waitress
        checks.append(("waitress WSGI 服务器", True))
    except ImportError:
        checks.append(("waitress WSGI 服务器", False))

    print("=" * 50)
    print("MASS 部署环境检查")
    print("=" * 50)
    all_ok = True
    for name, ok in checks:
        status = "OK" if ok else "MISSING"
        symbol = "[OK]" if ok else "[X]"
        print(f"  {symbol} {name}: {status}")
        if not ok:
            all_ok = False
    print("=" * 50)

    return all_ok


def start_server(host="0.0.0.0", port=5000, threads=4):
    """启动 waitress 生产服务器"""
    from app import create_app

    app = create_app()

    logger.info(f"MASS v2.1 生产服务启动于 http://{host}:{port}")
    logger.info(f"WSGI 服务器: waitress (threads={threads})")
    logger.info(f"DEBUG 模式: {app.config.get('FLASK_DEBUG', False)}")
    logger.info(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'mock')}")
    logger.info(f"USE_MOCK_LLM: {app.config.get('USE_MOCK_LLM', True)}")

    import waitress
    waitress.serve(
        app,
        host=host,
        port=port,
        threads=threads,
        clear_untrusted_proxy_headers=True,
        ident="MASS/2.1",
    )


def stop_server():
    """查找并停止 MASS 服务进程"""
    import psutil

    stopped = False
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "deploy.py" in cmdline and "python" in proc.info.get("name", "").lower():
                print(f"终止进程 PID={proc.info['pid']}: {cmdline[:80]}")
                proc.terminate()
                stopped = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if stopped:
        print("MASS 服务已停止")
    else:
        print("未找到运行中的 MASS 服务")


def main():
    parser = argparse.ArgumentParser(description="MASS 生产部署脚本")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="监听端口 (默认: 5000)")
    parser.add_argument("--threads", type=int, default=4, help="工作线程数 (默认: 4)")
    parser.add_argument("--stop", action="store_true", help="停止运行中的服务")
    parser.add_argument("--check", action="store_true", help="仅检查环境，不启动")
    args = parser.parse_args()

    if args.stop:
        try:
            stop_server()
        except ImportError:
            print("请先安装 psutil: pip install psutil")
            sys.exit(1)
        return

    if not check_environment():
        print("\n环境检查未通过，请修复上述问题后再部署。")
        sys.exit(1)

    if args.check:
        print("\n环境检查通过，可以部署。")
        return

    print(f"\n正在启动 MASS 生产服务...")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  线程: {args.threads}")
    print(f"  按 Ctrl+C 停止服务\n")

    start_server(host=args.host, port=args.port, threads=args.threads)


if __name__ == "__main__":
    main()
