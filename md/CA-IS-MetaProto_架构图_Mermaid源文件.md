# CA-IS-MetaProto 架构图（Mermaid 源文件）

> 可复制粘贴到 https://mermaid.live 或 Markdown 渲染器中查看

---

## 图1：训练阶段完整架构

```mermaid
flowchart TB
    subgraph data["一、数据采集 & 标注（一次性成本）"]
        A1[工业相机<br/>原始图像采集]
        A2[ROI 裁剪<br/>缺陷类别标注]
        A3[域标签分配<br/>材料/光照/工艺]
        A4[结构化数据<br/>domain_X/class/specimen]

        A1 --> A2 --> A3 --> A4
    end

    subgraph domain["二、域划分 & 工况建模（一次性成本）"]
        B1[确定训练域<br/>0,1,2 + 测试域 3]
        B2[构建 7D 工况向量<br/>亮度/对比度/噪声<br/>模糊/色偏/视角/纹理]
        B3[ConditionEncoder<br/>7D → 128D embedding]

        B1 --> B2 --> B3
    end

    subgraph p0["三、Phase 0 预训练（一次性 GPU，1~4h）"]
        C1[多域混合 ROI 数据<br/>pretrain/]
        C2[PretrainMultitaskModel<br/>YOLO26 backbone<br/>+ 多任务 Head]
        C3[SupCon Loss<br/>+ CE Loss]
        C4["pretrain_best.pt<br/>通用视觉特征"]

        C1 --> C2 --> C3 --> C4
    end

    subgraph p1["四、Phase 1 元学习训练（一次性 GPU，2~8h）"]
        D1[pretrain_best.pt<br/>Backbone 权重]
        D2[AviationDefectModel<br/>Backbone + FiLM + ProtoNet]

        D2 --> D3["Episode 采样器"]
        D3 --> D3a["同域 Episode<br/>稳定学习基础判别"]
        D3 --> D3b["跨域 Episode<br/>逼迫迁移能力<br/>curriculum 0.2→0.4→0.8"]

        D2 --> D4["DomainReliability<br/>MLP → 0~1 可靠性权重<br/>自适应 Loss 加权"]

        D2 --> D5["ProtoNet Head<br/>trim_mean 原型<br/>Mahalanobis 距离"]

        D1 --> D2
    end

    subgraph export["五、模型导出（一次性，30min）"]
        E1[best.pt]
        E2[模型量化/优化<br/>TorchScript / ONNX]
        E3[部署包<br/>权重 + 配置 + API]

        E1 --> E2 --> E3
    end

    data --> domain
    domain --> p0
    p0 --> p1
    p1 --> export

    style data fill:#e8f5e9,stroke:#2e7d32
    style domain fill:#fff3e0,stroke:#ef6c00
    style p0 fill:#e3f2fd,stroke:#1565c0
    style p1 fill:#f3e5f5,stroke:#7b1fa2
    style export fill:#fce4ec,stroke:#c2185b
```

---

## 图2：部署推理阶段完整架构

```mermaid
flowchart TB
    subgraph init["一、新缺陷上线（5 张样本 + 工况输入）"]
        F1[new 缺陷样本<br/>5 张标注图像]
        F2[7D 工况向量输入<br/>当前产线参数]
        F3["Support Set<br/>5 张 × N 类 → 原型"]
    end

    subgraph deploy["二、Few-Shot 适配（~10 秒，无 GPU 需求）"]
        G1[加载 best.pt<br/>本地推理引擎]
        G2["输入 Support 图像<br/>→ Backbone + FiLM]
        G3["trim_mean 原型构建<br/>同类样本平均（去极值）"]
        G4["原型存储<br/>in-memory"]
    end

    subgraph infer["三、本地推理（工业现场实时）"]
        H1[Query 图像输入<br/>工业相机实时采集]
        H2["Backbone + FiLM<br/>提取 Query 特征"]
        H3["计算 Mahalanobis 距离<br/>到各类原型"]
        H4["分类结果输出<br/>缺陷类别 + 置信度"]
        H5[触发后续动作<br/>分拣/复检/报警]
    end

    subgraph expand["四、持续扩展（按需）"]
        I1[新产线/新产品上线]
        I2[提供新 5 张样本]
        I3[更新 Support Set<br/>原型自动刷新]
        I4[无需重新训练<br/>无需 GPU]
    end

    subgraph superiority["vs. 传统方案对比"]
        J1[传统方案<br/>每换缺陷需重训<br/>大量同分布样本<br/>GPU 持续消耗]
        J2[CA-IS-MetaProto<br/>5 张即上线<br/>单卡 8GB 本地推理<br/>数据不出厂]

        style J1 fill:#ffebee,stroke:#c62828
        style J2 fill:#e8f5e9,stroke:#2e7d32
    end

    F1 --> F3
    F2 --> G2
    F3 --> G4

    F1 --> G2
    G1 --> G2
    G2 --> G3 --> G4

    G4 --> H1
    H1 --> H2 --> H3 --> H4 --> H5

    H5 --> I1
    I1 --> I2 --> I3 --> G4

    style F1 fill:#e8f5e9,stroke:#2e7d32
    style F2 fill:#fff3e0,stroke:#ef6c00
    style F3 fill:#e8f5e9,stroke:#2e7d32
    style deploy fill:#e3f2fd,stroke:#1565c0
    style infer fill:#f3e5f5,stroke:#7b1fa2
    style expand fill:#fff8e1,stroke:#f9a825
```

---

## 图3：训练阶段关键创新点串联（简化版）

```mermaid
flowchart LR
    subgraph Input["数据输入层"]
        I1[工业图像]
        I2[7D 工况向量]
    end

    subgraph Core["核心架构"]
        B[Backbone<br/>ResNet18 / YOLO26]
        C[ConditionEncoder<br/>7D → 128D]
        F[FiLM 调制层<br/>γ ⊙ F + β]
        P[ProtoNet Head<br/>trim_mean + Mahal.]
    end

    subgraph Innovation["三大创新点"]
        IN1["① 工况感知<br/>Condition + FiLM"]
        IN2["② 跨域 Episode<br/>Curriculum 学习"]
        IN3["③ 域可靠性加权<br/>DomainReliability"]
    end

    subgraph Output["输出"]
        O1[原型向量]
        O2[缺陷分类结果]
    end

    I1 --> B
    I2 --> C
    C --> F
    B --> F
    F --> P
    P --> O1 --> O2

    IN1 --> C
    IN1 --> F
    IN2 --> P
    IN3 --> P

    style Core fill:#e3f2fd,stroke:#1565c0
    style Innovation fill:#fff8e1,stroke:#f9a825
    style Output fill:#e8f5e9,stroke:#2e7d32
```

---

*文件版本：v1.0 | 生成日期：2026-09-12 | 可在 https://mermaid.live 预览*
