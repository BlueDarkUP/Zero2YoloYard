/**
 * PoseAnnotator - Manual Keypoint / Skeleton Annotator Plugin
 *
 * 设计目标：完全不写死点位数量或名称。每个"骨架实例"就是一个
 * AnnotationObject { type:'keypoint', label, bbox, keypoints:[{name,x,y,v}, ...] }，
 * keypoints 数组长度、顺序、命名完全自由 —— 可以是 COCO-17，也可以是 6 点的机械臂，
 * 21 点的手部，或者任何你想标的东西。
 *
 * "点位模板 (Schema)" 只是纯前端的标注效率工具（存 localStorage，按 视频+类别 区分），
 * 用来一键把某个类别常用的点位一次性放到画布上，而不是逼你手动敲字。它不是数据格式的
 * 一部分 —— 完全不用模板、每次现场点、现场起名字，一样能正常标注、显示、保存。
 *
 * 与其它 Annotator（SegmentationAnnotator / ClassificationAnnotator）保持同样的接口约定：
 * onMouseDown / onMouseMove / onMouseUp / onContextMenu / render(ctx, annotations, selectedId)
 */
class PoseAnnotator {
    constructor(core) {
        this.core = core;

        // 当前选中的骨架实例内部、被选中的关键点下标（null = 未选中任何点，只选中了整个实例）
        this.selectedKeypointIndex = null;
        this.draggedKeypointIndex = null;
        this.isDraggingPoint = false;

        // "放置新骨架" armed 状态：点了 Add Skeleton 之后，等待用户在画布上点一下决定放在哪
        this.isPlacingNewInstance = false;
        this.armedClass = null;

        // 无模板兜底：逐点点击 + 现场起名的模式
        this.isFreeformAdding = false;
        this.activeNewInstance = null;

        // 悬停/预览用
        this.hoverPoint = null;

        // 内联命名输入框（用原生 DOM，避免用阻塞式 prompt()）
        this.nameInputEl = null;

        this.initDOM();
        this.bindShortcuts();
        this.updateSidebarList();
    }

    // ============================================================
    // Schema (点位模板) —— 全局持久化与数据库存储
    // ============================================================

    schemaStorageKey(label) {
        return `zyy_pose_schema_global::${label}`;
    }

    getDefaultCOCO17Schema() {
        return {
            points: [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ],
            edges: [
                ["right_ankle", "right_knee"], ["right_knee", "right_hip"], ["left_ankle", "left_knee"],
                ["left_knee", "left_hip"], ["right_hip", "left_hip"], ["right_shoulder", "right_hip"],
                ["left_shoulder", "left_hip"], ["right_shoulder", "left_shoulder"], ["right_shoulder", "right_elbow"],
                ["left_shoulder", "left_elbow"], ["right_elbow", "right_wrist"], ["left_elbow", "left_wrist"],
                ["right_eye", "left_eye"], ["nose", "right_eye"], ["nose", "left_eye"],
                ["right_eye", "right_ear"], ["left_eye", "left_ear"]
            ]
        };
    }

    getSchema(label) {
        if (!label) label = 'person';

        // 1. 优先从内存全局缓存（从后端 DB 预加载）获取
        if (window.classKeypointSchemas && window.classKeypointSchemas[label]) {
            const s = typeof window.classKeypointSchemas[label] === 'string'
                ? JSON.parse(window.classKeypointSchemas[label])
                : window.classKeypointSchemas[label];
            if (s && Array.isArray(s.points) && s.points.length > 0) {
                return { points: s.points, edges: Array.isArray(s.edges) ? s.edges : [] };
            }
        }

        // 2. 从 localStorage（优先全局 key，其次旧的 video 绑定 key）获取
        try {
            const raw = localStorage.getItem(this.schemaStorageKey(label)) ||
                        localStorage.getItem(`zyy_pose_schema::${this.core.videoUuid}::${label}`);
            if (raw) {
                const parsed = JSON.parse(raw);
                if (parsed && Array.isArray(parsed.points) && parsed.points.length > 0) {
                    return { points: parsed.points, edges: Array.isArray(parsed.edges) ? parsed.edges : [] };
                }
            }
        } catch (e) { /* ignore */ }

        // 3. 无自定义模板时，默认回退到标准 COCO-17 骨架连线模板
        return this.getDefaultCOCO17Schema();
    }

    saveSchema(label, schema) {
        if (!label) return;
        const schemaData = {
            points: schema.points || [],
            edges: schema.edges || []
        };

        if (!window.classKeypointSchemas) window.classKeypointSchemas = {};
        window.classKeypointSchemas[label] = schemaData;

        try {
            localStorage.setItem(this.schemaStorageKey(label), JSON.stringify(schemaData));
            localStorage.setItem(`zyy_pose_schema::${this.core.videoUuid}::${label}`, JSON.stringify(schemaData));
        } catch (e) { /* ignore */ }

        // 持久化保存至后端 SQLite 数据库
        $.ajax({
            url: '/api/saveClassKeypointSchema',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ label: label, schema: schemaData }),
            success: function(res) {
                if (!res.success) {
                    console.warn("Failed to persist keypoint schema to database:", res.message);
                }
            }
        });
    }

    // ============================================================
    // 选中类别（复用与其它模式一致的 Class Registry 逻辑）
    // ============================================================

    getSelectedClass() {
        if (this.core.selectedClass) return this.core.selectedClass;
        if (typeof window.activeClass !== 'undefined' && window.activeClass) return window.activeClass;
        const activeLi = $('#class-list li.active');
        if (activeLi.length) {
            const cls = activeLi.data('class-name');
            if (cls) { this.core.selectedClass = cls; return cls; }
        }
        return null;
    }

    getColorForClass(label) {
        if (typeof window.stringToColor === 'function') return window.stringToColor(label);
        return '#00f0ff';
    }

    // ============================================================
    // DOM / 侧边栏初始化
    // ============================================================

    initDOM() {
        const self = this;

        $(document).off('click', '#btn-add-skeleton').on('click', '#btn-add-skeleton', function () {
            const cls = self.getSelectedClass();
            if (!cls) {
                if (typeof window.showToast === 'function') {
                    window.showToast('⚠️ 请先在右侧 Class Registry 选择/创建一个类别', 3000);
                }
                return;
            }
            self.isPlacingNewInstance = true;
            self.armedClass = cls;
            self.isFreeformAdding = false;
            if (typeof window.showToast === 'function') {
                window.showToast(`🎯 点击画布任意位置放置 [${cls}] 骨架`, 2500);
            }
        });

        $(document).off('click', '#btn-edit-pose-schema').on('click', '#btn-edit-pose-schema', function () {
            const cls = self.getSelectedClass();
            if (!cls) {
                if (typeof window.showToast === 'function') {
                    window.showToast('⚠️ 请先在右侧 Class Registry 选择/创建一个类别', 3000);
                }
                return;
            }
            self.openSchemaEditor(cls);
        });

        $(document).off('click', '#btn-finish-pose-freeform').on('click', '#btn-finish-pose-freeform', function () {
            self.finishFreeformAdding();
        });

        // 关节可见性按钮（Visible / Occluded / Absent）
        $(document).off('click', '.task-pose [data-v]').on('click', '.task-pose [data-v]', function () {
            const v = parseInt($(this).data('v'));
            self.setSelectedKeypointVisibility(v);
        });

        this.updateVisibilityButtonsUI();
    }

    bindShortcuts() {
        const self = this;
        window.addEventListener('keydown', (e) => {
            if ($(e.target).is('input, textarea')) return;

            if (e.key === 'Escape') {
                if (self.nameInputEl) { self.cancelNameInput(); return; }
                if (self.isPlacingNewInstance) {
                    self.isPlacingNewInstance = false;
                    self.armedClass = null;
                    if (typeof window.showToast === 'function') window.showToast('已取消放置', 1200);
                    return;
                }
                if (self.isFreeformAdding) {
                    self.finishFreeformAdding();
                    return;
                }
                if (self.selectedKeypointIndex !== null) {
                    self.selectedKeypointIndex = null;
                    self.updateVisibilityButtonsUI();
                    self.core.render();
                    return;
                }
                if (self.core.selectedObjectId) {
                    self.core.selectedObjectId = null;
                    self.updateSidebarList();
                    self.core.render();
                }
                return;
            }

            if (e.key === 'Enter') {
                if (self.nameInputEl) { self.confirmNameInput(); return; }
                if (self.isFreeformAdding) { self.finishFreeformAdding(); return; }
            }

            if (e.key === 'Delete' || e.key === 'Backspace') {
                self.deleteSelection();
                return;
            }

            // 0 / 1 / 2 直接设置选中关键点的可见性（COCO 惯例：0=不存在 1=遮挡 2=可见）
            if (['0', '1', '2'].includes(e.key) && self.selectedKeypointIndex !== null) {
                e.preventDefault();
                self.setSelectedKeypointVisibility(parseInt(e.key));
            }
        });
    }

    // ============================================================
    // 内联命名输入框（无模板时，逐点现场起名）
    // ============================================================

    showNameInput(screenX, screenY, defaultName) {
        this.cancelNameInput();
        const input = document.createElement('input');
        input.type = 'text';
        input.value = defaultName || '';
        input.placeholder = '关键点名称，如 nose / joint_1 ...';
        input.className = 'pose-inline-name-input';
        input.style.left = `${screenX}px`;
        input.style.top = `${screenY}px`;
        document.body.appendChild(input);
        input.focus();
        input.select();

        const self = this;
        input.addEventListener('keydown', (e) => {
            e.stopPropagation();
            if (e.key === 'Enter') { e.preventDefault(); self.confirmNameInput(); }
            else if (e.key === 'Escape') { e.preventDefault(); self.cancelNameInput(); }
        });
        input.addEventListener('blur', () => { self.confirmNameInput(); });

        this.nameInputEl = input;
    }

    confirmNameInput() {
        if (!this.nameInputEl) return;
        const val = this.nameInputEl.value.trim();
        const obj = this.activeNewInstance;
        if (obj && obj.keypoints.length > 0) {
            const lastPt = obj.keypoints[obj.keypoints.length - 1];
            lastPt.name = val || `point_${obj.keypoints.length}`;
        }
        this.nameInputEl.remove();
        this.nameInputEl = null;
        this.recomputeBbox(obj);
        this.core.saveAnnotations();
        this.updateSidebarList();
        this.core.render();
    }

    cancelNameInput() {
        if (!this.nameInputEl) return;
        // 取消命名 = 撤销刚刚加的这个点
        const obj = this.activeNewInstance;
        if (obj && obj.keypoints.length > 0) {
            obj.keypoints.pop();
        }
        this.nameInputEl.remove();
        this.nameInputEl = null;
        this.recomputeBbox(obj);
        this.core.render();
    }

    finishFreeformAdding() {
        if (this.nameInputEl) this.confirmNameInput();
        const obj = this.activeNewInstance;
        if (obj && obj.keypoints.length === 0) {
            // 什么都没加，直接把这个空实例删掉
            const idx = this.core.annotations.objects.findIndex(o => o.id === obj.id);
            if (idx >= 0) this.core.annotations.objects.splice(idx, 1);
            this.core.selectedObjectId = null;
        }
        this.isFreeformAdding = false;
        this.activeNewInstance = null;
        $('#btn-finish-pose-freeform').hide();
        this.core.saveAnnotations();
        this.updateSidebarList();
        this.core.render();
        if (typeof window.showToast === 'function') window.showToast('✅ 骨架标注完成', 1500);
    }

    // ============================================================
    // 增删改
    // ============================================================

    recomputeBbox(obj) {
        if (!obj || !obj.keypoints || obj.keypoints.length === 0) { if (obj) obj.bbox = null; return; }
        const visible = obj.keypoints.filter(k => (k.v ?? 2) > 0);
        const pts = visible.length > 0 ? visible : obj.keypoints;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        pts.forEach(p => {
            minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
            minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
        });
        const pad = 12;
        const imgW = (this.core.image && this.core.image.naturalWidth) || Infinity;
        const imgH = (this.core.image && this.core.image.naturalHeight) || Infinity;
        obj.bbox = [
            Math.max(0, minX - pad),
            Math.max(0, minY - pad),
            Math.min(imgW, maxX + pad),
            Math.min(imgH, maxY + pad)
        ];
    }

    setSelectedKeypointVisibility(v) {
        const obj = this.getSelectedObject();
        if (!obj || this.selectedKeypointIndex === null) return;
        const kp = obj.keypoints[this.selectedKeypointIndex];
        if (!kp) return;
        kp.v = v;
        this.core.saveAnnotations();
        this.updateVisibilityButtonsUI();
        this.updateSidebarList();
        this.core.render();
    }

    getSelectedObject() {
        if (!this.core.selectedObjectId) return null;
        return (this.core.annotations.objects || []).find(o => o.id === this.core.selectedObjectId) || null;
    }

    deleteSelection() {
        const obj = this.getSelectedObject();
        if (obj && this.selectedKeypointIndex !== null) {
            // 只删单个关键点
            obj.keypoints.splice(this.selectedKeypointIndex, 1);
            this.selectedKeypointIndex = null;
            this.recomputeBbox(obj);
            this.core.saveAnnotations();
            this.updateVisibilityButtonsUI();
            this.updateSidebarList();
            this.core.render();
        } else if (this.core.selectedObjectId) {
            // 删整个骨架实例
            const idx = (this.core.annotations.objects || []).findIndex(o => o.id === this.core.selectedObjectId);
            if (idx >= 0) {
                this.core.annotations.objects.splice(idx, 1);
                this.core.selectedObjectId = null;
                this.core.saveAnnotations();
                this.updateSidebarList();
                this.core.render();
            }
        }
    }

    materializeInstance(pt, cls) {
        const schema = this.getSchema(cls);
        const obj = {
            id: 'pose_' + Date.now() + '_' + Math.floor(Math.random() * 1000),
            type: 'keypoint',
            label: cls,
            bbox: null,
            keypoints: []
        };

        if (schema.points.length > 0) {
            const cols = Math.max(1, Math.ceil(Math.sqrt(schema.points.length)));
            schema.points.forEach((name, i) => {
                const dx = (i % cols) * 26 - ((cols - 1) * 13);
                const dy = Math.floor(i / cols) * 26 - ((Math.ceil(schema.points.length / cols) - 1) * 13);
                obj.keypoints.push({ name: name, x: pt.x + dx, y: pt.y + dy, v: 2 });
            });
            this.core.annotations.objects.push(obj);
            this.recomputeBbox(obj);
            this.core.selectedObjectId = obj.id;
            this.selectedKeypointIndex = null;
            this.isPlacingNewInstance = false;
            this.armedClass = null;
            this.core.saveAnnotations();
            this.updateSidebarList();
            this.core.render();
            if (typeof window.showToast === 'function') {
                window.showToast('📍 已按模板放置，拖拽各点到正确位置', 2500);
            }
        } else {
            this.core.annotations.objects.push(obj);
            this.core.selectedObjectId = obj.id;
            this.selectedKeypointIndex = null;
            this.isPlacingNewInstance = false;
            this.armedClass = null;
            this.isFreeformAdding = true;
            this.activeNewInstance = obj;
            $('#btn-finish-pose-freeform').show();
            this.updateSidebarList();
            this.core.render();
            if (typeof window.showToast === 'function') {
                window.showToast('✏️ 未设置模板：依次点击画布添加关键点，Enter/右键/"完成"按钮结束', 3500);
            }
        }
    }

    // ============================================================
    // 鼠标事件（AnnotationCore 插件接口）
    // ============================================================

    hitTestKeypoint(pt, radiusPx) {
        const objects = this.core.annotations.objects || [];
        const r = radiusPx / this.core.zoom;
        let best = null, bestDist = Infinity;
        for (const obj of objects) {
            if (obj.type !== 'keypoint' || !obj.keypoints) continue;
            for (let i = 0; i < obj.keypoints.length; i++) {
                const kp = obj.keypoints[i];
                const d = Math.hypot(pt.x - kp.x, pt.y - kp.y);
                if (d <= r && d < bestDist) {
                    bestDist = d;
                    best = { obj, index: i };
                }
            }
        }
        return best;
    }

    onMouseDown(pt, e) {
        if (e.button === 1 || e.button === 2 || this.core.isSpacePressed) return;

        // 命名输入框弹出期间，画布点击一律忽略（先确认/取消命名）
        if (this.nameInputEl) return;

        if (this.isFreeformAdding && this.activeNewInstance) {
            this.activeNewInstance.keypoints.push({ name: '', x: pt.x, y: pt.y, v: 2 });
            this.core.render();
            this.showNameInput(e.clientX + 10, e.clientY + 10, `point_${this.activeNewInstance.keypoints.length}`);
            return;
        }

        if (this.isPlacingNewInstance && this.armedClass) {
            this.materializeInstance(pt, this.armedClass);
            return;
        }

        const hit = this.hitTestKeypoint(pt, 10);
        if (hit) {
            this.core.selectedObjectId = hit.obj.id;
            this.selectedKeypointIndex = hit.index;
            this.isDraggingPoint = true;
            this.draggedKeypointIndex = hit.index;
            this.updateVisibilityButtonsUI();
            this.updateSidebarList();
            this.core.render();
            return;
        }

        // 空白处点击：取消选中
        if (this.core.selectedObjectId || this.selectedKeypointIndex !== null) {
            this.core.selectedObjectId = null;
            this.selectedKeypointIndex = null;
            this.updateVisibilityButtonsUI();
            this.updateSidebarList();
            this.core.render();
        }
    }

    onMouseMove(pt, e) {
        this.hoverPoint = pt;

        if (this.isDraggingPoint && this.draggedKeypointIndex !== null) {
            const obj = this.getSelectedObject();
            if (obj && obj.keypoints[this.draggedKeypointIndex]) {
                obj.keypoints[this.draggedKeypointIndex].x = pt.x;
                obj.keypoints[this.draggedKeypointIndex].y = pt.y;
                this.core.render();
            }
            return;
        }

        if (this.isPlacingNewInstance || this.isFreeformAdding) {
            this.core.render();
        }
    }

    onMouseUp(pt, e) {
        if (this.isDraggingPoint) {
            this.isDraggingPoint = false;
            const obj = this.getSelectedObject();
            if (obj) {
                this.recomputeBbox(obj);
                this.core.saveAnnotations();
                this.updateSidebarList();
            }
            this.draggedKeypointIndex = null;
        }
    }

    onContextMenu(pt, e) {
        if (this.isFreeformAdding) { this.finishFreeformAdding(); return; }
        const hit = this.hitTestKeypoint(pt, 10);
        if (hit) {
            hit.obj.keypoints.splice(hit.index, 1);
            if (this.core.selectedObjectId === hit.obj.id && this.selectedKeypointIndex === hit.index) {
                this.selectedKeypointIndex = null;
            }
            this.recomputeBbox(hit.obj);
            this.core.saveAnnotations();
            this.updateVisibilityButtonsUI();
            this.updateSidebarList();
            this.core.render();
        }
    }

    // ============================================================
    // 渲染
    // ============================================================

    colorForVisibility(v) {
        if (v === 0) return 'rgba(160,160,170,0.9)';   // absent - 灰
        if (v === 1) return '#ffb400';                  // occluded - 橙
        return '#00ff88';                                // visible - 绿
    }

    render(ctx, annotations, selectedId) {
        const objects = (annotations.objects || []).filter(o => o.type === 'keypoint');
        const zoom = this.core.zoom;

        objects.forEach(obj => {
            const isSelectedInstance = obj.id === selectedId;
            const classColor = this.getColorForClass(obj.label);
            const kps = obj.keypoints || [];
            const byName = {};
            kps.forEach(k => { if (k.name) byName[k.name] = k; });

            // 骨架连线（按该类别的模板 edges，用名字匹配；双端都存在且都不是 absent 才画）
            const schema = this.getSchema(obj.label);
            if (schema.edges && schema.edges.length > 0) {
                ctx.save();
                ctx.strokeStyle = classColor;
                ctx.lineWidth = (isSelectedInstance ? 2.5 : 1.5) / zoom;
                ctx.globalAlpha = isSelectedInstance ? 0.95 : 0.55;
                schema.edges.forEach(([a, b]) => {
                    const pa = byName[a], pb = byName[b];
                    if (pa && pb && (pa.v ?? 2) > 0 && (pb.v ?? 2) > 0) {
                        ctx.beginPath();
                        ctx.moveTo(pa.x, pa.y);
                        ctx.lineTo(pb.x, pb.y);
                        ctx.stroke();
                    }
                });
                ctx.restore();
            }

            // 关键点
            kps.forEach((kp, idx) => {
                const v = kp.v ?? 2;
                const isSelectedPoint = isSelectedInstance && idx === this.selectedKeypointIndex;
                const radius = (isSelectedPoint ? 7 : 5) / zoom;

                ctx.save();
                ctx.beginPath();
                ctx.arc(kp.x, kp.y, radius, 0, Math.PI * 2);
                ctx.fillStyle = this.colorForVisibility(v);
                ctx.globalAlpha = v === 0 ? 0.55 : (isSelectedInstance ? 1.0 : 0.85);
                ctx.fill();
                ctx.lineWidth = (isSelectedPoint ? 2.5 : 1.2) / zoom;
                ctx.strokeStyle = isSelectedPoint ? '#ffffff' : 'rgba(0,0,0,0.6)';
                ctx.globalAlpha = 1.0;
                ctx.stroke();

                if (isSelectedInstance) {
                    ctx.font = `${11 / zoom}px "JetBrains Mono", monospace`;
                    ctx.fillStyle = '#ffffff';
                    ctx.globalAlpha = 0.9;
                    ctx.fillText(kp.name || `#${idx}`, kp.x + radius + 3 / zoom, kp.y - radius - 3 / zoom);
                }
                ctx.restore();
            });
        });

        // 放置模式下的准心预览
        if ((this.isPlacingNewInstance || this.isFreeformAdding) && this.hoverPoint) {
            ctx.save();
            ctx.strokeStyle = 'rgba(0, 240, 255, 0.9)';
            ctx.lineWidth = 1.5 / zoom;
            const s = 10 / zoom;
            ctx.beginPath();
            ctx.moveTo(this.hoverPoint.x - s, this.hoverPoint.y);
            ctx.lineTo(this.hoverPoint.x + s, this.hoverPoint.y);
            ctx.moveTo(this.hoverPoint.x, this.hoverPoint.y - s);
            ctx.lineTo(this.hoverPoint.x, this.hoverPoint.y + s);
            ctx.stroke();
            ctx.restore();
        }

        this.updateSidebarList();
    }

    // ============================================================
    // 侧边栏骨架实例列表
    // ============================================================

    updateVisibilityButtonsUI() {
        const obj = this.getSelectedObject();
        const kp = (obj && this.selectedKeypointIndex !== null) ? obj.keypoints[this.selectedKeypointIndex] : null;
        const currentV = kp ? (kp.v ?? 2) : null;
        $('.task-pose [data-v]').each(function () {
            const v = parseInt($(this).data('v'));
            $(this).toggleClass('active', currentV !== null && v === currentV);
        });
        $('.task-pose [data-v]').prop('disabled', currentV === null);
    }

    updateSidebarList() {
        const countSpan = document.getElementById('pose-count');
        const listDiv = document.getElementById('pose-object-list');
        if (!listDiv) return;

        const objects = (this.core.annotations.objects || []).filter(o => o.type === 'keypoint');
        if (countSpan) countSpan.textContent = objects.length;

        listDiv.innerHTML = '';
        const self = this;

        objects.forEach((obj, idx) => {
            const isSelected = obj.id === this.core.selectedObjectId;
            const wrapper = document.createElement('div');
            wrapper.className = `mb-2 p-2 border rounded ${isSelected ? 'border-primary' : 'border-secondary'}`;
            wrapper.style.background = isSelected ? 'rgba(0,240,255,0.08)' : 'rgba(255,255,255,0.02)';

            const header = document.createElement('div');
            header.className = 'd-flex justify-content-between align-items-center';
            header.style.cursor = 'pointer';

            const name = document.createElement('span');
            name.className = 'font-weight-bold small';
            const visibleCount = (obj.keypoints || []).filter(k => (k.v ?? 2) > 0).length;
            name.textContent = `#${idx + 1} [${obj.label}] · ${visibleCount}/${(obj.keypoints || []).length} pts`;
            header.appendChild(name);

            const delBtn = document.createElement('button');
            delBtn.className = 'btn btn-sm btn-danger py-0 px-2';
            delBtn.innerHTML = '<i class="bi bi-trash"></i>';
            delBtn.onclick = (e) => {
                e.stopPropagation();
                const i = self.core.annotations.objects.findIndex(o => o.id === obj.id);
                if (i >= 0) self.core.annotations.objects.splice(i, 1);
                if (self.core.selectedObjectId === obj.id) { self.core.selectedObjectId = null; self.selectedKeypointIndex = null; }
                self.core.saveAnnotations();
                self.core.render();
            };
            header.appendChild(delBtn);

            header.onclick = () => {
                self.core.selectedObjectId = obj.id;
                self.selectedKeypointIndex = null;
                self.updateVisibilityButtonsUI();
                self.updateSidebarList();
                self.core.render();
            };
            wrapper.appendChild(header);

            if (isSelected) {
                const ptList = document.createElement('div');
                ptList.className = 'mt-2';
                (obj.keypoints || []).forEach((kp, kpIdx) => {
                    const row = document.createElement('div');
                    const isPtSelected = self.selectedKeypointIndex === kpIdx;
                    row.className = `d-flex justify-content-between align-items-center small py-1 px-2 mb-1 rounded ${isPtSelected ? 'bg-info text-dark' : ''}`;
                    row.style.cursor = 'pointer';

                    const vBadge = { 0: '⚪', 1: '🟠', 2: '🟢' }[kp.v ?? 2] || '🟢';
                    const label = document.createElement('span');
                    label.textContent = `${vBadge} ${kp.name || ('point_' + kpIdx)}`;
                    row.appendChild(label);

                    const delPtBtn = document.createElement('i');
                    delPtBtn.className = 'bi bi-x-lg text-danger';
                    delPtBtn.style.cursor = 'pointer';
                    delPtBtn.onclick = (e) => {
                        e.stopPropagation();
                        obj.keypoints.splice(kpIdx, 1);
                        if (self.selectedKeypointIndex === kpIdx) self.selectedKeypointIndex = null;
                        self.recomputeBbox(obj);
                        self.core.saveAnnotations();
                        self.updateVisibilityButtonsUI();
                        self.updateSidebarList();
                        self.core.render();
                    };
                    row.appendChild(delPtBtn);

                    row.onclick = () => {
                        self.selectedKeypointIndex = kpIdx;
                        self.updateVisibilityButtonsUI();
                        self.updateSidebarList();
                        self.core.render();
                    };

                    ptList.appendChild(row);
                });
                wrapper.appendChild(ptList);
            }

            listDiv.appendChild(wrapper);
        });
    }

    // ============================================================
    // 点位模板编辑器（可选的效率工具，不是标注数据的一部分）
    // ============================================================

    openSchemaEditor(label) {
        $('#pose-schema-modal').remove();

        const schema = this.getSchema(label);
        const self = this;

        const modal = $(`
            <div id="pose-schema-modal" class="keybind-modal-backdrop">
                <div class="keybind-modal-card" style="width: 480px; text-align: left; max-height: 82vh; overflow-y: auto;">
                    <div class="keybind-modal-header">
                        <span>✏️ 点位模板 — <span class="target-class-pill" style="font-size:0.8rem;">${label}</span></span>
                        <button type="button" class="close text-light ml-auto" id="pose-schema-close-x">&times;</button>
                    </div>
                    <div class="small text-muted mb-2">
                        这个模板只影响以后点"Add Skeleton"时自动放置哪些点，不是唯一格式限制——
                        标注时随时可以手动加点/删点，数量完全不受限制。
                    </div>

                    <div class="d-flex justify-content-between align-items-center mt-3 mb-1">
                        <b class="small text-uppercase text-info">点位列表 (Points)</b>
                        <button class="btn btn-xs btn-outline-success" id="pose-schema-load-coco17">载入 COCO-17 示例模板</button>
                    </div>
                    <div id="pose-schema-points-list"></div>
                    <div class="input-group input-group-sm mt-2">
                        <input type="text" id="pose-schema-new-point" class="form-control tool-input" placeholder="新点位名称，如 head / hinge_a ...">
                        <div class="input-group-append">
                            <button class="btn btn-sm btn-outline-info" id="pose-schema-add-point">+ 添加</button>
                        </div>
                    </div>

                    <div class="mt-3 mb-1"><b class="small text-uppercase text-info">骨架连线 (Bones，可选)</b></div>
                    <div id="pose-schema-edges-list"></div>
                    <div class="input-group input-group-sm mt-2">
                        <select id="pose-schema-edge-a" class="form-control tool-input"></select>
                        <select id="pose-schema-edge-b" class="form-control tool-input"></select>
                        <div class="input-group-append">
                            <button class="btn btn-sm btn-outline-info" id="pose-schema-add-edge">+ 连接</button>
                        </div>
                    </div>

                    <div class="mt-3 mb-1"><b class="small text-uppercase text-info">导入 / 导出 JSON</b></div>
                    <textarea id="pose-schema-json" class="form-control tool-input" rows="4" style="font-family: monospace; font-size: 0.75rem;"></textarea>
                    <div class="mt-1">
                        <button class="btn btn-xs btn-outline-secondary" id="pose-schema-apply-json">应用上方 JSON</button>
                    </div>

                    <div class="keybind-modal-footer">
                        <button class="btn btn-sm btn-secondary mr-2" id="pose-schema-cancel">关闭</button>
                        <button class="btn btn-sm btn-primary" id="pose-schema-save">保存模板</button>
                    </div>
                </div>
            </div>
        `);
        $('body').append(modal);

        let working = { points: [...schema.points], edges: schema.edges.map(e => [...e]) };

        function renderPoints() {
            const list = $('#pose-schema-points-list');
            list.empty();
            if (working.points.length === 0) {
                list.append('<div class="text-muted small">暂无点位，下面手动添加，或载入示例模板。</div>');
            }
            working.points.forEach((name, i) => {
                const row = $(`
                    <div class="d-flex align-items-center mb-1">
                        <span class="badge badge-secondary mr-2">${i + 1}</span>
                        <input type="text" class="form-control form-control-sm tool-input mr-2" value="${name}">
                        <i class="bi bi-x-lg text-danger" style="cursor:pointer;"></i>
                    </div>
                `);
                row.find('input').on('change', function () {
                    const oldName = working.points[i];
                    const newName = $(this).val().trim() || oldName;
                    working.edges.forEach(e => {
                        if (e[0] === oldName) e[0] = newName;
                        if (e[1] === oldName) e[1] = newName;
                    });
                    working.points[i] = newName;
                    renderEdgeSelectors();
                    renderEdges();
                    syncJsonBox();
                });
                row.find('i').on('click', function () {
                    const removed = working.points[i];
                    working.points.splice(i, 1);
                    working.edges = working.edges.filter(e => e[0] !== removed && e[1] !== removed);
                    renderPoints();
                    renderEdgeSelectors();
                    renderEdges();
                    syncJsonBox();
                });
                list.append(row);
            });
        }

        function renderEdgeSelectors() {
            const a = $('#pose-schema-edge-a').empty();
            const b = $('#pose-schema-edge-b').empty();
            working.points.forEach(p => {
                a.append(`<option value="${p}">${p}</option>`);
                b.append(`<option value="${p}">${p}</option>`);
            });
        }

        function renderEdges() {
            const list = $('#pose-schema-edges-list');
            list.empty();
            if (working.edges.length === 0) {
                list.append('<div class="text-muted small">暂无连线（连线只影响显示效果，不影响标注数据）。</div>');
            }
            working.edges.forEach((edge, i) => {
                const row = $(`
                    <div class="d-flex justify-content-between align-items-center small py-1">
                        <span>${edge[0]} — ${edge[1]}</span>
                        <i class="bi bi-x-lg text-danger" style="cursor:pointer;"></i>
                    </div>
                `);
                row.find('i').on('click', function () {
                    working.edges.splice(i, 1);
                    renderEdges();
                    syncJsonBox();
                });
                list.append(row);
            });
        }

        function syncJsonBox() {
            $('#pose-schema-json').val(JSON.stringify(working, null, 2));
        }

        renderPoints();
        renderEdgeSelectors();
        renderEdges();
        syncJsonBox();

        $('#pose-schema-add-point').on('click', function () {
            const input = $('#pose-schema-new-point');
            const val = input.val().trim();
            if (!val) return;
            if (working.points.includes(val)) {
                if (typeof window.showToast === 'function') window.showToast('⚠️ 已存在同名点位', 2000);
                return;
            }
            working.points.push(val);
            input.val('');
            renderPoints();
            renderEdgeSelectors();
            syncJsonBox();
        });
        $('#pose-schema-new-point').on('keydown', function (e) {
            if (e.key === 'Enter') { e.preventDefault(); $('#pose-schema-add-point').click(); }
        });

        $('#pose-schema-add-edge').on('click', function () {
            const a = $('#pose-schema-edge-a').val();
            const b = $('#pose-schema-edge-b').val();
            if (!a || !b || a === b) return;
            const exists = working.edges.some(e => (e[0] === a && e[1] === b) || (e[0] === b && e[1] === a));
            if (exists) return;
            working.edges.push([a, b]);
            renderEdges();
            syncJsonBox();
        });

        $('#pose-schema-load-coco17').on('click', function () {
            working = {
                points: ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle"],
                edges: [["right_ankle", "right_knee"], ["right_knee", "right_hip"], ["left_ankle", "left_knee"],
                    ["left_knee", "left_hip"], ["right_hip", "left_hip"], ["right_shoulder", "right_hip"],
                    ["left_shoulder", "left_hip"], ["right_shoulder", "left_shoulder"], ["right_shoulder", "right_elbow"],
                    ["left_shoulder", "left_elbow"], ["right_elbow", "right_wrist"], ["left_elbow", "left_wrist"],
                    ["right_eye", "left_eye"], ["nose", "right_eye"], ["nose", "left_eye"],
                    ["right_eye", "right_ear"], ["left_eye", "left_ear"]]
            };
            renderPoints();
            renderEdgeSelectors();
            renderEdges();
            syncJsonBox();
            if (typeof window.showToast === 'function') window.showToast('已载入 COCO-17 示例（还没保存，可继续编辑）', 2000);
        });

        $('#pose-schema-apply-json').on('click', function () {
            try {
                const parsed = JSON.parse($('#pose-schema-json').val());
                if (!Array.isArray(parsed.points)) throw new Error('points 必须是数组');
                working = { points: parsed.points, edges: Array.isArray(parsed.edges) ? parsed.edges : [] };
                renderPoints();
                renderEdgeSelectors();
                renderEdges();
                if (typeof window.showToast === 'function') window.showToast('JSON 已应用（还没保存）', 1800);
            } catch (e) {
                if (typeof window.showToast === 'function') window.showToast('⚠️ JSON 格式错误：' + e.message, 3000);
            }
        });

        $('#pose-schema-save').on('click', () => {
            self.saveSchema(label, working);
            if (typeof window.showToast === 'function') window.showToast(`✅ [${label}] 模板已保存（${working.points.length} 点）`, 2000);
            modal.remove();
            self.core.render();
        });
        $('#pose-schema-cancel, #pose-schema-close-x').on('click', () => modal.remove());
    }
}
