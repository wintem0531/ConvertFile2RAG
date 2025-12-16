#!/usr/bin/env python3
"""
RAG-Anything 依赖安装脚本

该脚本用于安装 RAG-Anything 及其依赖项。
"""

import os
import subprocess
import sys


def run_command(command, description):
    """运行命令并处理输出"""
    print(f"\n{'=' * 60}")
    print(f"执行: {description}")
    print(f"命令: {command}")
    print(f"{'=' * 60}")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.stdout:
        print("标准输出:")
        print(result.stdout)

    if result.stderr:
        print("错误输出:")
        print(result.stderr)

    if result.returncode != 0:
        print(f"命令执行失败，退出码: {result.returncode}")
        return False
    else:
        print(f"✅ {description} 执行成功")
        return True


def main():
    print("RAG-Anything 依赖安装脚本")
    print("=" * 60)

    # 显示 Python 版本信息
    print(f"Python 版本: {sys.version}")

    # 安装基础 RAG-Anything
    success = run_command("uv add raganything", "安装基础 RAG-Anything")

    if not success:
        print("❌ 基础 RAG-Anything 安装失败，尝试使用 pip 安装")
        success = run_command("pip install raganything", "使用 pip 安装基础 RAG-Anything")
        if not success:
            print("❌ RAG-Anything 安装失败，请手动安装")
            return False

    # 安装扩展依赖
    success = run_command("uv add 'raganything[all]'", "安装 RAG-Anything 扩展依赖")

    if not success:
        print("❌ 扩展依赖安装失败，尝试使用 pip 安装")
        success = run_command("pip install 'raganything[all]'", "使用 pip 安装 RAG-Anything 扩展依赖")

    # 检查 MinerU 安装
    print("\n检查 MinerU 安装...")
    try:
        result = subprocess.run("mineru --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ MinerU 已安装: {result.stdout.strip()}")
        else:
            print("❌ MinerU 未安装或配置有问题")
            print("请按照 RAG-Anything 文档安装 MinerU")
            print("参考: https://github.com/opendatalab/MinerU/blob/master/README_zh-CN.md")
    except Exception as e:
        print(f"❌ 检查 MinerU 时出错: {str(e)}")

    # 检查 LibreOffice 安装
    print("\n检查 LibreOffice 安装...")
    try:
        result = subprocess.run("libreoffice --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ LibreOffice 已安装: {result.stdout.strip()}")
        else:
            print("⚠️ LibreOffice 未安装")
            print("Office 文档处理需要 LibreOffice")
            print("请从 https://www.libreoffice.org/download/download/ 下载安装")
    except Exception as e:
        print(f"❌ 检查 LibreOffice 时出错: {str(e)}")
        print("⚠️ LibreOffice 未安装")
        print("Office 文档处理需要 LibreOffice")
        print("请从 https://www.libreoffice.org/download/download/ 下载安装")

    print("\n" + "=" * 60)
    print("安装脚本执行完成")
    print("=" * 60)

    return True


if __name__ == "__main__":
    main()
