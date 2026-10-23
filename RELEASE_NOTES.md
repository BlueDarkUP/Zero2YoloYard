# Zero2YoloYard v2.2.0 Release Notes

> 对比上一版本 **v2.1.0** 的主要更新与变更说明。

---

### 一、 核心 AI 模型与推理加速
- **SAM 3.0 接入与单框极速响应**：引入开放词汇感知与全图盲扫定位 (TRUE LAM)；重构算法去除冗余跨模型编码，单框响应时间由 **2.33s 优化至 0.06s**（EXE 实测 0.16s），原生输出高精度多边形轮廓。
- **GKDT Grounded Pose 姿态估计**：集成 8.87 亿参数大模型，首创通过文本提示驱动自定义关键点 Schema（人体、机械臂、竞赛道具等）的零样本骨骼预测。
- **SAM 2.1 连续视频多目标追踪**：升级 Hiera 骨架，引入双向亚像素 PyLK 光流与 Top-Down 动力学外推，支持步进追踪、全视频批量追踪与 SSE 实时流式进度推送。
- **CLIP 视觉特征检索**：支持基于图像特征向量的无监督聚类与一键批量分类打标。

---

### 二、 标注范式全矩阵扩展
- **4 大任务范式完整闭环**：目标检测 (BBox/OBB 插值)、实例/语义分割 (多边形/涂抹)、图像分类 (1-9 键盲打)、姿态估计 (骨骼关节点插值)。
- **分割涂抹工具升级**：新增画笔与橡皮擦，重构 Moore-Neighbor 轮廓追踪算法，根除旧版内部交叉线与空洞 Bug。

---

### 三、 9 大工业级数据集导出器
- **9 种格式一键导出**：YOLOv8 Detect、COCO Detection、Pascal VOC、YOLO Segmentation、Semantic Mask (8-bit PNG)、YOLO Classification、Folder Classification、YOLO Pose、COCO Pose。
- **数据集质检中心**：支持分布直方图统计、CLIP 离群点筛查、标注一致性检查及数据增强实时预览。

---

### 四、 工程发布与安装部署
- **轻量独立 EXE (`Zero2YoloYard.exe`)**：PyInstaller 本地脱机编译，权重严格物理外部化；修复 PyTorch、TensorFlow Lite 与 OpenMP 运行时冲突；支持 EdgeChromium GUI 与 `--no-gui` 无头模式。
- **标准安装包 (`Zero2YoloYard_Installer_v2.0.exe`)**：Inno Setup 单文件安装器，自动配置工作区目录读写权限（`users-full`）、桌面/开始菜单快捷方式及控制面板规范卸载入口。

---

### 五、 质量验证与测试覆盖
- **路由覆盖率**：全部 81 个 Flask 路由 100% 自动化测试通过，零 500 崩溃，异常输入规范返回 400 提示。
- **函数覆盖率**：底层核心模块（`database.py`、`file_storage.py`、`bbox_writer.py`、`settings_manager.py`、`local_model_manager.py`）共 107 个函数 100% 测试通过。
- **端到端验证**：4 大 AI 基础模型与 9 大数据集导出器均在发布 EXE 上通过实测。

---

### 六、 代码变动统计 (Diff from v2.1.0)
- **提交跨度**：`4e88438 (v2.1.0) ... 6113d27 (HEAD)`（94 Commits）
- **变动文件**：762 个文件（`+556,274 / -4,125` 行）
- **主要新增**：`exporters/`（9 款导出器）、`sam3/`、`gkdt_engine/`、`installer/`、`clip_model.py`。
