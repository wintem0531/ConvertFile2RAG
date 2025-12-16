#!/usr/bin/env python3
"""
RAG-Anything 测试模块入口文件

运行方式:
1. python -m modes.RAGAnything
2. python modes/RAGAnything/__main__.py
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RAG-Anything 测试模块")

    parser.add_argument(
        "--action",
        choices=["install", "test-basic", "test-advanced", "example", "test-doc"],
        default="test-basic",
        help=(
            "执行的操作: install(安装依赖), test-basic(基本功能测试), "
            "test-advanced(高级功能测试), example(使用示例), test-doc(特定文档测试)"
        ),
    )

    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API 密钥")

    parser.add_argument(
        "--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="OpenAI API 基础 URL"
    )

    return parser.parse_args()


async def run_install():
    """运行安装脚本"""
    print("运行依赖安装脚本...")

    try:
        import subprocess

        # 获取当前脚本目录
        script_dir = Path(__file__).parent
        install_script = script_dir / "install_dependencies.py"

        # 执行安装脚本
        result = subprocess.run([sys.executable, str(install_script)], capture_output=True, text=True)

        # 输出结果
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            print("✅ 依赖安装完成")
        else:
            print("❌ 依赖安装失败")

    except Exception as e:
        print(f"❌ 运行安装脚本失败: {str(e)}")


async def run_test_basic():
    """运行基本功能测试"""
    print("运行基本功能测试...")

    try:
        from test_basic_functionality import main as test_main

        await test_main()
    except ImportError as e:
        print(f"❌ 导入测试模块失败: {str(e)}")
    except Exception as e:
        print(f"❌ 基本功能测试失败: {str(e)}")


async def run_test_advanced(api_key, base_url):
    """运行高级功能测试"""
    print("运行高级功能测试...")

    try:
        # 设置环境变量
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url

        from test_advanced_functionality import main as test_main

        await test_main()
    except ImportError as e:
        print(f"❌ 导入测试模块失败: {str(e)}")
    except Exception as e:
        print(f"❌ 高级功能测试失败: {str(e)}")


async def run_example(api_key, base_url):
    """运行使用示例"""
    print("运行使用示例...")

    try:
        # 设置环境变量
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url

        from example_usage import main as example_main

        await example_main()
    except ImportError as e:
        print(f"❌ 导入示例模块失败: {str(e)}")
    except Exception as e:
        print(f"❌ 使用示例运行失败: {str(e)}")


async def run_test_document(api_key, base_url):
    """运行特定文档测试"""
    print("运行特定文档测试...")

    try:
        # 设置环境变量
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url

        from test_specific_document import main as test_doc_main

        await test_doc_main()
    except ImportError as e:
        print(f"❌ 导入特定文档测试模块失败: {str(e)}")
    except Exception as e:
        print(f"❌ 特定文档测试运行失败: {str(e)}")


async def main():
    """主函数"""
    args = parse_args()

    print("RAG-Anything 测试模块")
    print("=" * 50)

    if args.action == "install":
        await run_install()
    elif args.action == "test-basic":
        await run_test_basic()
    elif args.action == "test-advanced":
        if not args.api_key:
            print("⚠️ 未设置 API 密钥，高级功能测试可能会失败")
            print("设置方法:")
            print("1. 通过 --api-key 参数提供")
            print("2. 设置 OPENAI_API_KEY 环境变量")

        await run_test_advanced(args.api_key, args.base_url)
    elif args.action == "example":
        if not args.api_key:
            print("⚠️ 未设置 API 密钥，示例仅展示代码结构")
            print("设置方法:")
            print("1. 通过 --api-key 参数提供")
            print("2. 设置 OPENAI_API_KEY 环境变量")

        await run_example(args.api_key, args.base_url)
    elif args.action == "test-doc":
        if not args.api_key:
            print("⚠️ 未设置 API 密钥，部分测试可能会失败")
            print("设置方法:")
            print("1. 通过 --api-key 参数提供")
            print("2. 设置 OPENAI_API_KEY 环境变量")

        print("测试文档: test_file/input/齊系文字編.pdf (第20-25页)")

        await run_test_document(args.api_key, args.base_url)

    print("\n操作完成")


if __name__ == "__main__":
    asyncio.run(main())
