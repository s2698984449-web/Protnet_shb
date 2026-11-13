#!/usr/bin/env python3
"""批量下载论文 PDF。用法: python3 download_papers.py [目标目录]
默认目录: ~/缺陷检测论文
自动下载 arXiv / CVF / MDPI / Nature 等可直接取 PDF 的链接；
其余(IEEE、ScienceDirect、Springer、PMC、ResearchGate 等)写入 手动下载清单.txt。
"""
import os, re, ssl, subprocess, sys, time, urllib.request

PAPERS = [
    (1, 2017, 'Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks', 'https://arxiv.org/abs/1703.03400'),
    (2, 2017, 'Prototypical Networks for Few-shot Learning', 'https://arxiv.org/abs/1703.05175'),
    (3, 2018, 'Learning to Compare: Relation Network for Few-Shot Learning', 'https://arxiv.org/abs/1711.06025'),
    (4, 2019, 'Few-shot Object Detection via Feature Reweighting', 'https://arxiv.org/abs/1812.01866'),
    (5, 2020, 'Frustratingly Simple Few-Shot Object Detection', 'https://arxiv.org/abs/2003.06957'),
    (6, 2020, 'Prior Guided Feature Enrichment Network for Few-Shot Segmentation (PFENet)', 'https://arxiv.org/abs/2008.01449'),
    (7, 2021, 'Meta-DETR: Few-Shot Object Detection via Unified Image-Level Meta-Learning', 'https://arxiv.org/abs/2103.11731'),
    (8, 2021, 'Learning Unsupervised Metaformer for Anomaly Detection', 'https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Learning_Unsupervised_Metaformer_for_Anomaly_Detection_ICCV_2021_paper.pdf'),
    (9, 2022, 'Few-Shot Steel Surface Defect Detection', 'https://ieeexplore.ieee.org/abstract/document/9623623'),
    (10, 2023, 'XDNet: A Few-Shot Meta-Learning Approach for Cross-Domain Visual Inspection', 'https://openaccess.thecvf.com/content/CVPR2023W/VISION/html/Lee_XDNet_A_Few-Shot_Meta-Learning_Approach_for_Cross-Domain_Visual_Inspection_CVPRW_2023_paper.html'),
    (11, 2023, 'A Few-Shot Defect Detection Method for Transmission Lines Based on Meta-Attention and Feature Reconstruction (Meta PowerNet)', 'https://www.proquest.com/docview/2819278674'),
    (12, 2023, 'Permute-MAML: exploring industrial surface defect detection algorithms for few-shot learning', 'https://link.springer.com/article/10.1007/s40747-023-01219-9'),
    (13, 2023, 'Adaptive-MAML: Few-shot metal surface defects diagnosis based on model-agnostic meta-learning', 'https://www.sciencedirect.com/science/article/abs/pii/S0263224123011764'),
    (14, 2023, 'Few-Shot Steel Defect Detection Method Based on Improved Meta-Learning Model', 'https://ieeexplore.ieee.org/document/10270277/'),
    (15, 2023, 'Few-Shot Steel Plate Surface Defect Detection with Multi-Relation Aggregation and Adaptive Support Learning', 'https://www.jstage.jst.go.jp/article/isijinternational/63/10/63_ISIJINT-2023-118/_article'),
    (16, 2024, 'Few-Shot Semantic Segmentation of Strip Steel Surface Defects Based on Meta-Learning（基于元学习的带钢表面缺陷少样本语义分割）', 'https://cjournal.hep.com.cn/1005-3026/CN/PDF/Y2024/V45/I03/354'),
    (17, 2025, 'SM-FSOD: A Second-Order Meta-Learning Algorithm for Few-Shot PCB Defect Object Detection', 'https://www.mdpi.com/2079-9292/14/19/3863'),
    (18, 2025, 'Robust Anomaly Detection in Industrial Environments via Meta-Learning (RAD)', 'https://arxiv.org/pdf/2508.17789'),
    (19, 2026, 'BayPrAnoMeta: Bayesian Proto-MAML for Few-Shot Industrial Image Anomaly Detection', 'https://arxiv.org/abs/2601.19992'),
    (20, 2026, 'R2-Net: A Hybrid Meta-Learning Framework for Cross-Domain Few-Shot Industrial Defect Detection', 'https://www.fujipress.jp/jaciii/jc/jacii003000041138/?full=1'),
    (21, 2022, 'A New Knowledge Distillation Network for Incremental Few-Shot Surface Defect Detection', 'https://arxiv.org/pdf/2209.00519'),
    (22, 2022, 'Coarse-to-Fine Few-Shot Defect Recognition With Dynamic Weighting and Joint Metric', 'https://www.researchgate.net/publication/362169407_Coarse-to-Fine_Few-Shot_Defect_Recognition_with_Dynamic_Weighting_and_Joint_Metric'),
    (23, 2023, 'FaNet: Feature-aware network for few shot classification of strip steel surface defects', 'https://www.sciencedirect.com/science/article/pii/S0263224123000106'),
    (24, 2023, 'FS-RSDD: Few-Shot Rail Surface Defect Detection with Prototype Learning', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10536558/'),
    (25, 2024, 'ODNet: A High Real-Time Network Using Orthogonal Decomposition for Few-Shot Strip Steel Surface Defect Classification', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11280595/'),
    (26, 2024, 'Sparse cross-transformer network for surface defect detection', 'https://www.nature.com/articles/s41598-024-75680-y'),
    (27, 2025, 'Mean shift based prototypical network for steel surface anomaly recognition (MSPro-Net)', 'https://www.nature.com/articles/s41598-025-31529-6'),
    (28, 2025, 'Attention-guided few-shot learning for metal surface defect classification', 'https://www.researchgate.net/publication/396620643_Attention-guided_few-shot_learning_for_metal_surface_defect_classification'),
    (29, 2025, 'Crossed Wavelet Convolution Network for Few-Shot Defect Detection of Industrial Chips', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12298282/'),
    (30, 2026, 'A prototype-aligned multi-scale attention network for few-shot industrial surface defect detection (PA-MSAN)', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC13618933/'),
    (31, 2020, 'Few-Shot Defect Segmentation Leveraging Abundant Normal Training Samples Through Normal Background Regularization and Crop-and-Paste Operation', 'https://arxiv.org/pdf/2007.09438'),
    (32, 2022, 'Selective Prototype Network for Few-Shot Metal Surface Defect Segmentation', 'https://www.researchgate.net/publication/362638512_Selective_Prototype_Network_for_Few-Shot_Metal_Surface_Defect_Segmentation'),
    (33, 2024, 'LERENet: Eliminating Intra-class Differences for Metal Surface Defect Few-shot Semantic Segmentation', 'https://arxiv.org/pdf/2403.11122'),
    (34, 2025, 'Exploring Few-Shot Defect Segmentation in General Industrial Scenarios with Metric Learning and Vision Foundation Models', 'https://arxiv.org/abs/2502.01216'),
    (35, 2025, 'Attention-Enhanced Prototypical Learning for Few-Shot Infrastructure Defect Segmentation', 'https://arxiv.org/pdf/2510.05266'),
    (36, 2026, 'Few-shot cross-episode adaptive memory for metal surface defect semantic segmentation', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12891459/'),
    (37, 2024, 'Few-Shot Steel Defect Detection Based on a Fine-Tuned Network with Serial Multi-Scale Attention', 'https://mdpi.com/2076-3417/14/13/5823'),
    (38, 2025, 'DSG-FSOD：few-shot steel plate surface defect detection（题名见链接）', 'https://link.springer.com/doi/10.1007/s40747-025-02131-0'),
    (39, 2025, 'Adaptive Graph Neural Networks with Semi-Supervised Multi-Modal Fusion for Few-Shot Steel Strip Defect Detection', 'https://www.mdpi.com/2227-9717/13/11/3520'),
    (40, 2023, 'WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation', 'https://arxiv.org/abs/2303.14814'),
    (41, 2024, 'PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection', 'https://arxiv.org/abs/2404.05231'),
    (42, 2024, 'Cross-domain Multi-modal Few-shot Object Detection via Rich Text', 'https://arxiv.org/pdf/2403.16188'),
    (43, 2025, 'NexViTAD: Few-shot Unsupervised Cross-Domain Defect Detection via Vision Foundation Models and Multi-Task Learning', 'https://arxiv.org/pdf/2507.07579'),
    (44, 2025, 'Cross-domain Few-shot Object Detection with Multi-modal Textual Enrichment', 'https://arxiv.org/pdf/2502.16469'),
    (45, 2026, 'Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection', 'https://arxiv.org/pdf/2603.18541'),
    (46, 2026, 'DevPrompt: Deviation-Based Prompt Learning for One-Normal Shot Image Anomaly Detection', 'https://arxiv.org/html/2601.15453'),
    (47, 2026, 'Anchor and Adapt: Asymmetric Prompt Adaptation for Few-Shot Industrial Anomaly Detection', 'https://arxiv.org/html/2610.07016v1'),
    (48, 2024, 'Few-Shot Object Detection: Research Advances and Challenges', 'https://arxiv.org/html/2404.04799v1'),
    (49, 2025, 'A Survey on Foundation-Model-Based Industrial Defect Detection', 'https://arxiv.org/pdf/2502.19106'),
    (50, 2025, 'Few-shot defect detection in industrial scenarios: A comprehensive review of challenges, advances, and frontier trends', 'https://www.researchgate.net/publication/396103886_Few-shot_defect_detection_in_industrial_scenarios_A_comprehensive_review_of_challenges_advances_and_frontier_trends'),
]

def pdf_url(u):
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", u)
    if m: return f"https://arxiv.org/pdf/{m.group(1)}"
    if "openaccess.thecvf.com" in u:
        return u.replace("/html/", "/papers/").replace("_paper.html", "_paper.pdf")
    if "mdpi.com/" in u: return u.rstrip("/") + "/pdf"
    if "nature.com/articles/" in u: return u.rstrip("/") + ".pdf"
    return None

def _ctx():
    # 绕开 Windows 证书库(部分 conda 环境读取它会报 ASN1: NOT_ENOUGH_DATA)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None
CTX = _ctx()

def fetch(url, fn):
    if CTX is not None:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r, open(fn, "wb") as f:
            f.write(r.read())
    else:  # 没有 certifi 时用系统 curl(Windows 10+ 自带 curl.exe)
        subprocess.run(["curl", "-L", "-sS", "-f", "-A", "Mozilla/5.0", "-o", fn, url], check=True)
    with open(fn, "rb") as f:
        if not f.read(5).startswith(b"%PDF"):
            os.remove(fn); raise ValueError("返回内容不是 PDF")

def safe(s): return re.sub(r'[\\/:*?"<>|]', "_", s)[:70]

d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/缺陷检测论文")
os.makedirs(d, exist_ok=True)
manual = []
for n, y, t, u in PAPERS:
    pu = pdf_url(u)
    if not pu:
        manual.append(f"{n:02d} [{y}] {t}\n   {u}"); continue
    fn = os.path.join(d, f"{n:02d}_{y}_{safe(t)}.pdf")
    if os.path.exists(fn): print("已存在", fn); continue
    try:
        fetch(pu, fn)
        print("OK  ", n, t[:50]); time.sleep(1)
    except Exception as e:
        print("失败", n, t[:50], e); manual.append(f"{n:02d} [{y}] {t}\n   {u}")
with open(os.path.join(d, "手动下载清单.txt"), "w", encoding="utf8") as f:
    f.write("\n".join(manual))
print(f"\n完成。需手动下载 {len(manual)} 篇，见 {d}/手动下载清单.txt")