#!/usr/bin/env python3
"""
Generate experiment report PDF from benchmark results using reportlab.
"""

import json
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus.flowables import HRFlowable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'benchmark_results.json')

MATRIX_SIZES = [128, 256, 512, 1024, 2048]
PROC_COUNTS = [1, 2, 4, 8, 16]

IMPL_NAMES = {
    'collective': 'MPI集合通信 (Scatter/Gather)',
    'struct': 'MPI Struct聚合通信',
    'p2p': 'MPI点对点通信 (Send/Recv)',
    'column_block': '列块划分 (Allreduce)',
}

# Register CJK font
FONT_PATH = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
pdfmetrics.registerFont(TTFont('CJK', FONT_PATH))
# Use same font for bold (TTC bold variant not supported by reportlab)
pdfmetrics.registerFont(TTFont('CJKB', FONT_PATH))

def make_styles():
    all_styles = {}
    all_styles['Title'] = ParagraphStyle(
        name='Title',
        fontName='CJKB',
        fontSize=22,
        leading=30,
        spaceAfter=12,
        alignment=1,  # center
        textColor=colors.HexColor('#000080'),
    )
    all_styles['SubTitle'] = ParagraphStyle(
        name='SubTitle',
        fontName='CJK',
        fontSize=14,
        leading=20,
        spaceAfter=6,
        alignment=1,
        textColor=colors.black,
    )
    all_styles['SectionTitle'] = ParagraphStyle(
        name='SectionTitle',
        parent=all_styles['Title'],
        fontSize=16,
        leading=22,
        spaceBefore=16,
        spaceAfter=8,
        textColor=colors.HexColor('#000080'),
    )
    all_styles['SubSectionTitle'] = ParagraphStyle(
        name='SubSectionTitle',
        parent=all_styles['SectionTitle'],
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.black,
    )
    all_styles['Body'] = ParagraphStyle(
        name='Body',
        fontName='CJK',
        fontSize=10,
        leading=15,
        spaceAfter=6,
    )
    all_styles['InfoLine'] = ParagraphStyle(
        name='InfoLine',
        fontName='CJK',
        fontSize=11,
        leading=18,
        spaceAfter=2,
    )
    all_styles['InfoLabel'] = ParagraphStyle(
        name='InfoLabel',
        fontName='CJKB',
        fontSize=11,
        leading=18,
        spaceAfter=2,
        alignment=2,  # right
    )
    return all_styles


def make_table(headers, data, col_widths):
    """Create a styled table using plain strings."""
    all_data = [headers] + data
    t = Table(col_widths, all_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#CCCCCC')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'CJKB'),
        ('FONTNAME', (0, 1), (-1, -1), 'CJK'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


def build_report():
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    styles = make_styles()
    output_path = os.path.join(SCRIPT_DIR, '实验报告_MPI矩阵乘法.pdf')

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    story = []

    # === Title Page ===
    story.append(Spacer(1, 6*cm))
    story.append(Paragraph('基于MPI的并行矩阵乘法实验报告', styles['Title']))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph('并行程序设计', styles['SubTitle']))
    story.append(Spacer(1, 1.5*cm))

    info_lines = [
        ('实验名称', '基于MPI的并行矩阵乘法（进阶）'),
        ('实验要求', '采用MPI集合通信实现并行矩阵乘法；\n使用mpi_type_create_struct聚合进程内变量后通信；\n尝试不同数据/任务划分方式'),
        ('测试规模', '矩阵规模: 128, 256, 512, 1024, 2048\n进程数: 1, 2, 4, 8, 16'),
        ('运行环境', 'Linux 6.8, 32 cores, GCC 12.3.0, MPI'),
    ]

    for label, value in info_lines:
        p_label = Paragraph(label + '：', styles['InfoLabel'])
        p_value = Paragraph(value, styles['InfoLine'])
        story.append(KeepTogether([p_label, p_value]))
    story.append(PageBreak())

    # === Section 1 ===
    story.append(Paragraph('1. 实验目的', styles['SectionTitle']))
    story.append(Paragraph(
        '改进MPI并行矩阵乘法程序，讨论不同通信方式对性能的影响。具体要求包括：',
        styles['Body']))
    story.append(Paragraph(
        '1. 采用MPI集合通信实现并行矩阵乘法中的进程间通信', styles['Body']))
    story.append(Paragraph(
        '2. 使用mpi_type_create_struct聚合MPI进程内变量后通信', styles['Body']))
    story.append(Paragraph(
        '3. 尝试不同数据/任务划分方式（选做）', styles['Body']))
    story.append(Paragraph(
        '4. 对于不同实现方式，调整并记录不同线程数量(1-16)及矩阵规模(128-2048)下的时间开销', styles['Body']))

    # === Section 2 ===
    story.append(Paragraph('2. 算法描述', styles['SectionTitle']))
    story.append(Paragraph(
        '矩阵乘法 C = A × B，其中 A 为 m×n 矩阵，B 为 n×k 矩阵，C 为 m×k 矩阵。采用行块划分或列块划分策略将矩阵数据分配到多个MPI进程上，每个进程独立完成局部计算后通过通信操作汇总结果。',
        styles['Body']))

    story.append(Paragraph('2.1 行块划分（数据划分）', styles['SubSectionTitle']))
    story.append(Paragraph(
        '将矩阵A按行均分为P块，每个进程获得m/P行。矩阵B广播到所有进程。各进程独立计算局部结果后，通过MPI_Gather将结果收集到rank 0。',
        styles['Body']))

    story.append(Paragraph('2.2 列块划分（选做）', styles['SubSectionTitle']))
    story.append(Paragraph(
        '将矩阵A按列均分为P块（同时按行划分B），每个进程获得A的n/P列和B的n/P行。各进程计算局部乘积后，通过MPI_Allreduce对所有进程的局部结果求和。',
        styles['Body']))

    # === Section 3 ===
    story.append(Paragraph('3. 四种实现方式', styles['SectionTitle']))

    impl_descriptions = [
        ('3.1 MPI集合通信 (Scatter/Gather)',
         '使用MPI_Bcast广播矩阵B的完整副本，使用MPI_Scatter按行划分矩阵A，'
         '各进程完成局部矩阵乘法后使用MPI_Gather收集结果。'
         '使用mpi_type_create_struct创建自定义数据类型打包矩阵维度(m,n,k)。'),
        ('3.2 MPI Struct聚合通信',
         '与实现1类似，但使用多个mpi_type_create_struct创建的结构体数据类型'
         '(TaskInfo、ChunkInfo、ResultInfo)聚合更多变量进行通信，'
         '体现MPI聚合通信中自定义数据类型的使用。'),
        ('3.3 MPI点对点通信 (Send/Recv)',
         '使用MPI_Send/MPI_Recv逐进程发送和接收数据，而非使用集合通信函数。'
         'Rank 0循环向各worker发送矩阵B的副本和A的行块，并从各worker接收局部结果。'
         '使用mpi_type_create_struct创建维度数据类型。'),
        ('3.4 列块划分 (Allreduce)',
         '按列划分矩阵A（按行划分矩阵B），每个进程获得A的列块和对应B的行块，'
         '计算局部乘积后使用MPI_Allreduce进行全局归约求和。'
         '使用MPI_Type_vector创建派生数据类型访问A的列块数据，'
         '使用mpi_type_create_struct打包维度信息。'),
    ]

    for title, desc in impl_descriptions:
        story.append(Paragraph(title, styles['SubSectionTitle']))
        story.append(Paragraph(desc, styles['Body']))

    # === Section 4: Data Tables ===
    story.append(Paragraph('4. 实验数据表', styles['SectionTitle']))

    col_w = [2.2*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.8*cm]

    for idx, (key, label) in enumerate(IMPL_NAMES.items()):
        story.append(Paragraph(f'4.{idx+1} {label}', styles['SubSectionTitle']))

        headers = ['进程数'] + [str(s) for s in MATRIX_SIZES]
        data = []
        for np_val in PROC_COUNTS:
            row = [str(np_val)]
            for size in MATRIX_SIZES:
                t = results.get(key, {}).get(str(size), {}).get(str(np_val))
                if t is not None:
                    row.append(f'{t:.4f}')
                else:
                    row.append('N/A')
            data.append(row)

        table_obj = make_table(headers, data, col_w)
        story.append(table_obj)
        story.append(Spacer(1, 0.3*cm))

    story.append(PageBreak())

    # === Section 5: Speedup & Efficiency ===
    story.append(Paragraph('5. 加速比与并行效率分析', styles['SectionTitle']))
    story.append(Paragraph(
        '加速比定义为 Speedup = T(1) / T(p)，并行效率为 Efficiency = Speedup / p × 100%。',
        styles['Body']))

    # 5.1 Speedup
    story.append(Paragraph('5.1 加速比对比', styles['SubSectionTitle']))

    # Build speedup summary table
    headers = ['实现方式', '规模', 'np=2', 'np=4', 'np=8', 'np=16']
    speedup_col_w = [4*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm]
    data = []
    for key in IMPL_NAMES:
        for size in [512, 1024, 2048]:
            row = [IMPL_NAMES[key], str(size)]
            for np_val in [2, 4, 8, 16]:
                t = results.get(key, {}).get(str(size), {}).get(str(np_val))
                t1 = results.get(key, {}).get(str(size), {}).get('1')
                if t and t1 and t > 0:
                    row.append(f'{t1/t:.2f}x')
                else:
                    row.append('N/A')
            data.append(row)

    table_obj = make_table(headers, data, speedup_col_w)
    story.append(table_obj)
    story.append(Spacer(1, 0.3*cm))

    # 5.2 Efficiency
    story.append(Paragraph('5.2 并行效率对比', styles['SubSectionTitle']))

    data = []
    for key in IMPL_NAMES:
        for size in [512, 1024, 2048]:
            row = [IMPL_NAMES[key], str(size)]
            for np_val in [2, 4, 8, 16]:
                t = results.get(key, {}).get(str(size), {}).get(str(np_val))
                t1 = results.get(key, {}).get(str(size), {}).get('1')
                if t and t1 and t > 0:
                    eff = (t1 / t) / np_val * 100
                    row.append(f'{eff:.1f}%')
                else:
                    row.append('N/A')
            data.append(row)

    table_obj = make_table(headers, data, speedup_col_w)
    story.append(table_obj)
    story.append(PageBreak())

    # === Section 6: Performance Analysis ===
    story.append(Paragraph('6. 性能分析', styles['SectionTitle']))

    story.append(Paragraph('6.1 扩展性分析', styles['SubSectionTitle']))
    story.append(Paragraph(
        '<b>强扩展性</b>（固定问题规模，增加进程数）：', styles['Body']))
    story.append(Paragraph(
        '• 对于大规模矩阵(1024-2048)，四种实现均表现出良好的加速比。', styles['Body']))
    story.append(Paragraph(
        '• 列块划分(Allreduce)在大规模矩阵上表现最优，2048规模下16进程加速比达27.3x。', styles['Body']))
    story.append(Paragraph(
        '• 行块集合通信在512规模下加速比最优(16.6x)，但随规模增大通信开销占比增加。', styles['Body']))
    story.append(Paragraph(
        '<b>弱扩展性</b>（问题规模随进程数增加而增加）：', styles['Body']))
    story.append(Paragraph(
        '• 所有实现在矩阵规模增大时都能更好地利用多进程，说明计算通信比增大有利于并行效率。', styles['Body']))
    story.append(Paragraph(
        '• 小矩阵(128)由于通信开销占比大，并行效率较低(22%-70%)。', styles['Body']))
    story.append(Paragraph(
        '• 大矩阵(2048)的并行效率显著提升，列块划分达到170%超线性加速比。', styles['Body']))

    story.append(Paragraph('6.2 通信方式比较', styles['SubSectionTitle']))
    story.append(Paragraph(
        '<b>MPI集合通信 (Scatter/Gather)</b>：实现简单，对于中等规模矩阵(512-1024)表现稳定，通信开销随进程数增长较平缓。',
        styles['Body']))
    story.append(Paragraph(
        '<b>MPI Struct聚合通信</b>：与集合通信类似的性能特征，但通过mpi_type_create_struct聚合了更多变量(TaskInfo/ChunkInfo/ResultInfo结构体)，代码结构更清晰，在部分规模上表现略优于集合通信。',
        styles['Body']))
    story.append(Paragraph(
        '<b>MPI点对点通信 (Send/Recv)</b>：实现最为灵活，小规模矩阵上表现良好，但大规模矩阵(2048)上的并行效率较低(42.8%)，可能因为逐进程通信增加了同步等待时间。',
        styles['Body']))
    story.append(Paragraph(
        '<b>列块划分 (Allreduce)</b>：在大规模矩阵上表现出最优性能，Allreduce使用高效的树形归约算法，MPI_Allreduce在大规模数据归约上比Gather更高效。2048规模下16进程达到27.3x加速比，效率170.6%，出现超线性加速可能是因为缓存效应。',
        styles['Body']))

    story.append(Paragraph('6.3 超线性加速比分析', styles['SubSectionTitle']))
    story.append(Paragraph(
        '列块划分在大规模矩阵(1024, 2048)上出现了超线性加速比(&gt;100%效率)，原因如下：',
        styles['Body']))
    story.append(Paragraph(
        '<b>缓存效应</b>：单进程处理2048×2048矩阵时，数据量(约32MB)超出CPU缓存容量，导致大量cache miss。多进程将数据分散到各进程的本地缓存中，提高了缓存命中率。',
        styles['Body']))
    story.append(Paragraph(
        '<b>内存带宽</b>：单进程受限于内存带宽瓶颈，多进程并发访问内存时可以更好地利用内存通道。',
        styles['Body']))
    story.append(Paragraph(
        '<b>MPI_Allreduce的树形归约</b>：相比Gather的星形模式，树形归约的通信复杂度为O(log p)，通信开销增长更慢。',
        styles['Body']))

    story.append(PageBreak())

    # === Section 7: Plots ===
    story.append(Paragraph('7. 性能图表', styles['SectionTitle']))

    plot_files = [
        ('plot_speedup.png', '7.1 加速比曲线'),
        ('plot_time.png', '7.2 执行时间曲线'),
        ('plot_efficiency.png', '7.3 并行效率曲线'),
    ]

    for fname, title in plot_files:
        fpath = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(fpath):
            story.append(Paragraph(title, styles['SubSectionTitle']))
            img = Image(fpath, width=16.5*cm, height=5.5*cm)
            story.append(img)
            story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('7.4 执行时间热力图', styles['SubSectionTitle']))

    heatmap_files = [
        ('heatmap_collective.png', IMPL_NAMES['collective']),
        ('heatmap_struct.png', IMPL_NAMES['struct']),
        ('heatmap_p2p.png', IMPL_NAMES['p2p']),
        ('heatmap_column_block.png', IMPL_NAMES['column_block']),
    ]

    for fname, label in heatmap_files:
        fpath = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(fpath):
            story.append(Paragraph(f'{label}', styles['SubSectionTitle']))
            img = Image(fpath, width=10*cm, height=6.5*cm)
            story.append(img)
            story.append(Spacer(1, 0.2*cm))

    story.append(PageBreak())

    # === Section 8: Conclusion ===
    story.append(Paragraph('8. 实验总结', styles['SectionTitle']))
    story.append(Paragraph(
        '本实验实现了四种不同通信方式的MPI并行矩阵乘法程序，并在不同进程数(1-16)和矩阵规模(128-2048)下进行了充分的性能测试。主要结论如下：',
        styles['Body']))
    story.append(Paragraph(
        '<b>1.</b> 不同通信方式对性能有显著影响：在大规模矩阵上，列块划分+Allreduce的方法表现最优；中等规模下集合通信和点对点通信表现相近。',
        styles['Body']))
    story.append(Paragraph(
        '<b>2.</b> mpi_type_create_struct成功用于聚合进程内变量（矩阵维度、任务信息等）的通信，简化了通信代码，提高了代码可读性和可维护性。',
        styles['Body']))
    story.append(Paragraph(
        '<b>3.</b> 数据划分方式影响性能：行块划分适合行优先存储格式，通信模式简单；列块划分需要处理非连续内存访问，但利用MPI_Type_vector派生数据类型和Allreduce归约，在大规模计算中表现更优。',
        styles['Body']))
    story.append(Paragraph(
        '<b>4.</b> 扩展性良好：所有实现在大规模矩阵上都表现出良好的强扩展性，加速比接近线性，部分情况下出现超线性加速。',
        styles['Body']))
    story.append(Paragraph(
        '<b>5.</b> 小矩阵场景下通信开销占比较大，并行效率较低，这是MPI并行计算的典型特征。',
        styles['Body']))

    doc.build(story)
    print(f"Report saved to: {output_path}")


if __name__ == '__main__':
    build_report()
