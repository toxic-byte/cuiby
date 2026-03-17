# cuiby

崔博越的研究生学术资料仓库，包含学位论文、学术论文及相关研究材料。

## 📁 项目结构

```
cuiby/
├── cby_latex/                  # 硕士学位论文（LaTeX）
│   ├── main.tex                # 论文主文件
│   ├── front/                  # 封面与摘要
│   ├── body/                   # 正文章节
│   │   ├── chapter01.tex       # 第1章 绪论
│   │   ├── chapter02.tex       # 第2章 相关理论与技术
│   │   ├── chapter03.tex       # 第3章 噪声识别与偏好提取的多行为去噪推荐方法
│   │   └── chapter04.tex       # 第4章 多模态多行为去噪推荐方法
│   ├── back/                   # 结论、致谢、发表文章等
│   ├── figures/                # 论文图片
│   ├── reference.bib           # 参考文献
│   ├── hitszthesis.cls         # HITSZ 学位论文模板
│   └── hitszthesis.sty         # 自定义宏包
│
└── cby_relative/               # 相关学术资料
    ├── cby_MZSGO_bioinformatics/   # MZSGO 生物信息学论文
    ├── 崔博越中期报告/              # 中期报告
    └── 崔博越开题报告/              # 开题报告
```

## 🔧 编译方法

```bash
cd cby_latex

# 完整编译流程
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex

# 或使用 Makefile
make thesis

# 或使用 latexmk
latexmk -xelatex main.tex
```

## 📄 License

论文模板基于 [LPPL 1.3c](http://www.latex-project.org/lppl.txt) 许可证。
