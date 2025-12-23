#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI tool for ConvertFile2RAG - PDF processing with MinerU"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import necessary modules
from modes.mineru.mineru_util import (
    MinerUOutput,
    parse_pdf,
    parse_pdf_simple
)

app = typer.Typer(help="ConvertFile2RAG CLI工具 - 用于PDF处理的MinerU工具")
console = Console()


def print_statistics(stats: dict):
    """打印处理统计信息"""
    table = Table(title="处理结果统计")
    table.add_column("类型", style="cyan", no_wrap=True)
    table.add_column("数量", style="green")
    table.add_column("备注", style="yellow")

    table.add_row("总页数", str(stats.get("total_pages", 0)))
    table.add_row("文本块", str(stats.get("text_blocks", 0)))
    table.add_row("图片块", str(stats.get("image_blocks", 0)))
    table.add_row("表格块", str(stats.get("table_blocks", 0)))
    table.add_row("公式块", str(stats.get("equation_blocks", 0)))
    table.add_row("总块数", str(stats.get("total_blocks", 0)), style="bold green")

    console.print(table)


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    page_range: Optional[Tuple[int, int]] = None,
    backend: str = "vlm-mlx-engine",
    lang: str = "ch",
    debug: bool = False,
    **kwargs
) -> MinerUOutput:
    """处理单个PDF文件"""
    console.print(f"\n[bold blue]开始处理PDF:[/bold blue] {pdf_path.name}")

    if debug:
        console.print(f"[yellow]调试信息:[/yellow]")
        console.print(f"  - PDF路径: {pdf_path}")
        console.print(f"  - 输出目录: {output_dir}")
        console.print(f"  - 页面范围: {page_range or '全部'}")
        console.print(f"  - 后端引擎: {backend}")

    # 使用类化的MinerU解析器
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        task = progress.add_task(f"[cyan]解析PDF: {pdf_path.name}", total=None)

        try:
            # 使用标准化的parse_pdf函数，返回MinerUOutput对象
            output = parse_pdf(
                pdf_path=pdf_path,
                page_range=page_range,
                output_dir=output_dir,
                backend=backend,
                lang=lang,
                return_format="content_list",
                **kwargs
            )

            progress.update(task, completed=True)

        except Exception as e:
            console.print(f"[red]解析失败: {e}[/red]")
            raise

    # 计算统计信息
    stats = {
        "total_pages": len(set(block.page_idx for block in output.blocks)) + 1,
        "text_blocks": len(output.text_blocks),
        "image_blocks": len(output.image_blocks),
        "table_blocks": len(output.table_blocks),
        "equation_blocks": len(output.equation_blocks),
        "total_blocks": len(output.blocks)
    }

    console.print(f"[green]解析完成[/green]")
    print_statistics(stats)

    # 展示内容预览
    console.print("\n[bold]内容预览:[/bold]")

    # 展示文本块预览
    if output.text_blocks:
        console.print("\n[cyan]文本块示例:[/cyan]")
        for i, block in enumerate(output.text_blocks[:3]):
            preview = block.text[:100] + "..." if len(block.text) > 100 else block.text
            console.print(f"  {i+1}. {preview}")

    # 展示图片信息
    if output.image_blocks:
        console.print(f"\n[cyan]图片信息 (共{len(output.image_blocks)}张):[/cyan]")
        for i, block in enumerate(output.image_blocks[:3]):
            console.print(f"  {i+1}. {block.img_path}")
            if block.caption:
                console.print(f"     标题: {' '.join(block.caption[:50])}")

    # 保存所有内容
    console.print(f"\n[bold]保存结果到:[/bold] {output_dir}")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("保存内容...", total=None)

        # 保存所有内容（文本、Markdown、图片、结构化数据）
        output.save_all(output_dir, save_text=True, save_markdown=True)

        progress.update(task, completed=True)

    console.print("[green]保存完成[/green]")

    # 显示保存的文件
    console.print("\n[bold]生成的文件:[/bold]")
    files = list(output_dir.rglob("*"))
    for file in sorted(files):
        if file.is_file():
            relative_path = file.relative_to(output_dir)
            size = file.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size/(1024*1024):.1f}MB"
            console.print(f"  文件: {relative_path} [dim]({size_str})[/dim]")

    return output


@app.command()
def process(
    pdf_path: str = typer.Argument(..., help="PDF文件路径"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录（默认：PDF同目录）"),
    page_range: Optional[str] = typer.Option(None, "--pages", "-p", help="页面范围，例如：1-10 或 5"),
    backend: str = typer.Option("vlm-mlx-engine", "--backend", "-b", help="后端引擎 (vlm-mlx-engine 或 pipeline)"),
    lang: str = typer.Option("ch", "--lang", "-l", help="文档语言 (ch=中文, en=英文)"),
    debug: bool = typer.Option(False, "--debug", "-d", help="启用调试模式")
):
    """处理PDF文件并提取内容"""

    # 验证PDF文件
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        console.print(f"[red]错误：PDF文件不存在: {pdf_path}[/red]")
        raise typer.Exit(1)

    if not pdf_file.suffix.lower() == '.pdf':
        console.print(f"[red]错误：文件不是PDF格式: {pdf_path}[/red]")
        raise typer.Exit(1)

    # 设置输出目录
    if output_dir is None:
        output_path = pdf_file.parent / f"{pdf_file.stem}_processed"
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # 解析页面范围
    page_range_tuple = None
    if page_range:
        try:
            if '-' in page_range:
                start, end = map(int, page_range.split('-'))
                page_range_tuple = (start, end)
            else:
                page_num = int(page_range)
                page_range_tuple = (page_num, page_num)
        except ValueError:
            console.print(f"[red]错误：无效的页面范围格式: {page_range}[/red]")
            console.print("正确格式示例：1-10 或 5")
            raise typer.Exit(1)

    try:
        # 处理PDF
        output = process_single_pdf(
            pdf_path=pdf_file,
            output_dir=output_path,
            page_range=page_range_tuple,
            backend=backend,
            lang=lang,
            debug=debug
        )

        # 打印总结
        console.print(f"\n[bold green]处理完成！[/bold green]")
        console.print(f"结果保存在: {output_path}")
        console.print(f"共提取 {len(output.blocks)} 个内容块")

        # 展示一些可以进行的后续操作
        console.print("\n[bold]后续操作示例:[/bold]")
        console.print("import json")
        console.print(f"with open('{output_path}/structure.json', 'r', encoding='utf-8') as f:")
        console.print("    data = json.load(f)")
        console.print("    # 处理提取的数据...")

    except Exception as e:
        console.print(f"\n[red]处理失败: {e}[/red]")
        if debug:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="包含PDF文件的目录"),
    output_dir: str = typer.Argument(..., help="输出目录"),
    pattern: str = typer.Option("*.pdf", "--pattern", help="文件名匹配模式"),
    backend: str = typer.Option("vlm-mlx-engine", "--backend", "-b", help="后端引擎"),
    lang: str = typer.Option("ch", "--lang", "-l", help="文档语言"),
    parallel: bool = typer.Option(False, "--parallel", help="并行处理（实验性功能）")
):
    """批量处理目录中的所有PDF文件"""

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        console.print(f"[red]错误：输入目录不存在: {input_dir}[/red]")
        raise typer.Exit(1)

    # 查找所有PDF文件
    pdf_files = list(input_path.glob(pattern))

    if not pdf_files:
        console.print(f"[yellow]警告：在 {input_dir} 中没有找到匹配 {pattern} 的文件[/yellow]")
        raise typer.Exit(0)

    console.print(f"[green]找到 {len(pdf_files)} 个PDF文件[/green]")

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 处理文件
    success_count = 0
    failed_files = []

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]批量处理中...", total=len(pdf_files))

        for pdf_file in pdf_files:
            try:
                # 为每个PDF创建单独的输出目录
                pdf_output_dir = output_path / pdf_file.stem
                pdf_output_dir.mkdir(exist_ok=True)

                # 处理PDF
                process_single_pdf(
                    pdf_path=pdf_file,
                    output_dir=pdf_output_dir,
                    backend=backend,
                    lang=lang
                )

                success_count += 1

            except Exception as e:
                console.print(f"[red]处理 {pdf_file.name} 失败: {e}[/red]")
                failed_files.append((pdf_file.name, str(e)))

            progress.update(task, advance=1)

    # 打印总结
    console.print(f"\n[bold green]批量处理完成！[/bold green]")
    console.print(f"成功: {success_count} 个文件")
    console.print(f"失败: {len(failed_files)} 个文件")

    if failed_files:
        console.print("\n[red]失败的文件:[/red]")
        for filename, error in failed_files:
            console.print(f"  - {filename}: {error}")


@app.command()
def analyze(
    output_path: str = typer.Argument(..., help="处理后的输出目录路径")
):
    """分析处理结果"""

    path = Path(output_path)

    if not path.exists():
        console.print(f"[red]错误：目录不存在: {output_path}[/red]")
        raise typer.Exit(1)

    # 查找structure.json文件
    structure_file = path / "structure.json"
    if not structure_file.exists():
        console.print(f"[red]错误：找不到结构化数据文件: {structure_file}[/red]")
        raise typer.Exit(1)

    # 读取数据
    with open(structure_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 显示元数据
    metadata = data.get('metadata', {})
    console.print("[bold]元数据:[/bold]")
    for key, value in metadata.items():
        console.print(f"  {key}: {value}")

    # 分析页面
    pages = data.get('pages', [])
    console.print(f"\n[bold]页面分析 (共 {len(pages)} 页):[/bold]")

    total_blocks = 0
    type_counts = {}

    for page in pages:
        page_num = page.get('page_number', 0)
        blocks = page.get('blocks', [])
        total_blocks += len(blocks)

        console.print(f"\n  第 {page_num} 页 ({len(blocks)} 个块):")

        for block in blocks:
            block_type = block.get('type', 'unknown')
            type_counts[block_type] = type_counts.get(block_type, 0) + 1

            content = block.get('text', '')[:50]
            if block_type == 'image':
                content = block.get('img_path', '')

            console.print(f"    - {block_type}: {content}...")

    # 总体统计
    console.print(f"\n[bold]总体统计:[/bold]")
    console.print(f"  总块数: {total_blocks}")
    for block_type, count in sorted(type_counts.items()):
        console.print(f"  {block_type}: {count}")


@app.command()
def version():
    """显示版本信息"""
    console.print("[bold green]ConvertFile2RAG CLI[/bold green]")
    console.print("版本: 1.0.0")
    console.print("基于 MinerU 构建")


if __name__ == "__main__":
    app()