#!/usr/bin/env python3
"""
从论文PDF中提取指定页面的图片，用于替换第二章的TikZ配图
"""
import subprocess
import os

PAPER_DIR = "/Users/toxic/school/paper"
FIGURES_DIR = "/Users/toxic/school/cuiby/cby_latex/figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

# 使用 pdfimages 提取所有图片到临时目录
# 也可以使用 pdf2image 将特定页面转为图片

# 我们先用 pdfplumber 查看每个PDF的页面数和内容，找到对应的图的位置
import pdfplumber

pdfs_to_check = [
    ("A Survey of Zero-Shot Learning- Seings, Methods, and Applications.pdf", "zsl_survey"),
    ("A_Review_of_Generalized_Zero-Shot_Learning_Methods.pdf", "gzsl_review"),
    ("A Survey of Large Language Models.pdf", "llm_survey"),
    ("Evolutionary-scale prediction of atomic-level protein structure with a language model.pdf", "esm2"),
    ("Learning_Transferable_Visual_Models_From_Natural_Language.pdf", "clip"),
    ("Houlsby et al. \"Parameter-Efficient Transfer Learning for NLP\" (ICML, 2019).pdf", "adapter"),
    ("ProtTrans_Toward_Understanding_the_Language_of_Life_Through_Self-Supervised_Learning.pdf", "prottrans"),
    ("LoRA- Low-Rank Adaptation of Large Language Models.pdf", "lora"),
    ("History, Development, and Principles of Large Language Models—An Introductory Survey.pdf", "llm_history"),
]

for pdf_name, label in pdfs_to_check:
    pdf_path = os.path.join(PAPER_DIR, pdf_name)
    if os.path.exists(pdf_path):
        with pdfplumber.open(pdf_path) as pdf:
            print(f"\n{'='*60}")
            print(f"[{label}] {pdf_name}")
            print(f"Total pages: {len(pdf.pages)}")
            # Check first few pages for figures
            for i, page in enumerate(pdf.pages[:15]):
                text = page.extract_text() or ""
                # Look for figure captions
                lines = text.split('\n')
                for line in lines:
                    lower = line.lower().strip()
                    if lower.startswith('fig') or 'figure' in lower[:20]:
                        print(f"  Page {i+1}: {line[:120]}")
    else:
        print(f"NOT FOUND: {pdf_name}")
