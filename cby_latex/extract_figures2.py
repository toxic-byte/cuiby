#!/usr/bin/env python3
"""
使用PyMuPDF从论文PDF中提取特定页面的图片区域
"""
import os
import fitz  # PyMuPDF

PAPER_DIR = "/Users/toxic/school/paper"
FIGURES_DIR = "/Users/toxic/school/cuiby/cby_latex/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

DPI = 300
ZOOM = DPI / 72  # 72 is the default DPI

def extract_page_as_image(pdf_name, page_num, output_name, crop_box=None):
    """
    从PDF中提取某一页并保存为PNG图片
    page_num: 页码 (从1开始)
    crop_box: (left_pct, upper_pct, right_pct, lower_pct) 百分比
    """
    pdf_path = os.path.join(PAPER_DIR, pdf_name)
    if not os.path.exists(pdf_path):
        print(f"ERROR: {pdf_path} not found!")
        return False
    
    print(f"\nExtracting page {page_num} from {pdf_name}...")
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # 0-indexed
    
    mat = fitz.Matrix(ZOOM, ZOOM)
    pix = page.get_pixmap(matrix=mat)
    
    w, h = pix.width, pix.height
    print(f"  Page size: {w}x{h}")
    
    if crop_box:
        left = int(w * crop_box[0])
        upper = int(h * crop_box[1])
        right = int(w * crop_box[2])
        lower = int(h * crop_box[3])
        # Use irect for cropping
        clip_rect = fitz.Rect(
            page.rect.x0 + page.rect.width * crop_box[0],
            page.rect.y0 + page.rect.height * crop_box[1],
            page.rect.x0 + page.rect.width * crop_box[2],
            page.rect.y0 + page.rect.height * crop_box[3]
        )
        pix = page.get_pixmap(matrix=mat, clip=clip_rect)
        print(f"  Cropped to: {pix.width}x{pix.height}")
    
    output_path = os.path.join(FIGURES_DIR, output_name)
    pix.save(output_path)
    print(f"  Saved: {output_path}")
    doc.close()
    return True


# ========== 1. fig:zsl_framework - 零样本学习框架图 ==========
# 从 GZSL Review 论文 Fig.1 (Page 2) - ZSL vs GZSL schematic diagram
extract_page_as_image(
    "A_Review_of_Generalized_Zero-Shot_Learning_Methods.pdf",
    page_num=2,
    output_name="zsl_framework.png",
    crop_box=(0.05, 0.06, 0.95, 0.50)
)

# ========== 2. fig:zsl_evolution - ZSL技术演进/分类 ==========
# 从 GZSL Review 论文 Fig.7 (Page 6) - 方法分类体系 taxonomy
extract_page_as_image(
    "A_Review_of_Generalized_Zero-Shot_Learning_Methods.pdf",
    page_num=6,
    output_name="zsl_evolution.png",
    crop_box=(0.05, 0.55, 0.95, 0.98)
)

# ========== 3. fig:lm_evolution - 语言模型发展历程 ==========
# 从 History, Development, and Principles of LLM Fig.1 (Page 3)
extract_page_as_image(
    "History, Development, and Principles of Large Language Models—An Introductory Survey.pdf",
    page_num=3,
    output_name="lm_evolution.png",
    crop_box=(0.05, 0.05, 0.95, 0.62)
)

# ========== 4. fig:esm2 - 蛋白质语言模型 ==========
# 从 ProtTrans Fig.1 (Page 3) - Feature extraction overview
extract_page_as_image(
    "ProtTrans_Toward_Understanding_the_Language_of_Life_Through_Self-Supervised_Learning.pdf",
    page_num=3,
    output_name="esm2.png",
    crop_box=(0.05, 0.02, 0.95, 0.52)
)

# ========== 5. fig:contrastive - CLIP对比学习 ==========
# CLIP 论文 Figure 1 - 通常在第2页或第3页
extract_page_as_image(
    "Learning_Transferable_Visual_Models_From_Natural_Language.pdf",
    page_num=2,
    output_name="contrastive.png",
    crop_box=(0.05, 0.03, 0.95, 0.48)
)

# ========== 6. fig:adapter - 适配器架构 ==========
# Houlsby et al. Figure 2 (通常在第3页)
extract_page_as_image(
    'Houlsby et al. "Parameter-Efficient Transfer Learning for NLP" (ICML, 2019).pdf',
    page_num=3,
    output_name="adapter.png",
    crop_box=(0.05, 0.02, 0.95, 0.58)
)

print("\n\n===== All extractions complete! =====")
print(f"Figures saved to: {FIGURES_DIR}")

# List what we extracted
for f in sorted(os.listdir(FIGURES_DIR)):
    if f.endswith('.png') and f in ['zsl_framework.png', 'zsl_evolution.png', 'lm_evolution.png', 'esm2.png', 'contrastive.png', 'adapter.png']:
        full = os.path.join(FIGURES_DIR, f)
        size_kb = os.path.getsize(full) / 1024
        print(f"  {f}: {size_kb:.1f} KB")
