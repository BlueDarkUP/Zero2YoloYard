/**
 * AnnotationCore - Core Canvas State Manager, Zoom/Pan Engine, & Sidebar UI Sync
 */
class AnnotationCore {
    constructor(options) {
        this.canvas = options.canvas;
        this.ctx = this.canvas.getContext('2d');
        this.videoUuid = options.videoUuid;
        this.annotationType = options.annotationType || 'segmentation';

        this.currentFrame = 0;
        this.image = document.getElementById('frame-image') || new Image();

        // View Transform (Zoom & Pan)
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.isPanning = false;
        this.startPanX = 0;
        this.startPanY = 0;
        this.isSpacePressed = false;

        // Unified Annotation Data & Undo History
        this.annotations = {
            objects: [],
            classifications: []
        };
        this.history = [];
        this.historyIndex = -1;

        this.selectedObjectId = null;
        this.selectedClass = null;

        // Active Annotator Plugin
        this.annotator = null;

        this.initEvents();
        this.bindClassRegistry();
    }

    saveStateToHistory() {
        this.history = this.history.slice(0, this.historyIndex + 1);
        this.history.push(JSON.parse(JSON.stringify(this.annotations)));
        if (this.history.length > 50) {
            this.history.shift();
        }
        this.historyIndex = this.history.length - 1;
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.annotations = JSON.parse(JSON.stringify(this.history[this.historyIndex]));
            this.selectedObjectId = null;
            this.saveAnnotations(false);
            this.updateSidebarList();
            this.render();
            if (typeof window.showToast === 'function') {
                window.showToast('↩️ Undone', 1000);
            }
        }
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.annotations = JSON.parse(JSON.stringify(this.history[this.historyIndex]));
            this.selectedObjectId = null;
            this.saveAnnotations(false);
            this.updateSidebarList();
            this.render();
            if (typeof window.showToast === 'function') {
                window.showToast('↪️ Redone', 1000);
            }
        }
    }

    setAnnotator(annotatorInstance) {
        this.annotator = annotatorInstance;
    }

    getObjects() {
        return (this.annotations && Array.isArray(this.annotations.objects)) ? this.annotations.objects : [];
    }

    bindClassRegistry() {
        const self = this;
        // Listen for class selection from sidebar
        $(document).on('click', '#class-list .class-item-clickable', function() {
            setTimeout(() => {
                if (typeof activeClass !== 'undefined' && activeClass) {
                    self.selectedClass = activeClass;
                } else {
                    const li = $(this).closest('li');
                    self.selectedClass = li.data('class-name') || null;
                }
            }, 50);
        });
    }

    initEvents() {
        const container = this.canvas.parentElement;

        // Track Spacebar for Panning
        window.addEventListener('keydown', (e) => {
            if ($(e.target).is('input, textarea')) return;
            if (e.code === 'Space' && !this.isSpacePressed) {
                this.isSpacePressed = true;
                this.canvas.style.cursor = 'grab';
            }
        });

        window.addEventListener('keyup', (e) => {
            if (e.code === 'Space') {
                this.isSpacePressed = false;
                this.canvas.style.cursor = 'default';
            }
        });

        // Zoom (Wheel) - Centered on Mouse Cursor
        container.addEventListener('wheel', (e) => {
            e.preventDefault();
            const canvasContainer = document.getElementById('canvas-container');
            
            const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
            const newZoom = Math.max(0.1, Math.min(15.0, this.zoom * zoomFactor));

            // Adjust pan so point under cursor stays stationary relative to screen
            // mouse position relative to container
            const rect = canvasContainer.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            this.panX = this.panX - mouseX * (newZoom / this.zoom - 1);
            this.panY = this.panY - mouseY * (newZoom / this.zoom - 1);
            this.zoom = newZoom;

            canvasContainer.style.transformOrigin = '0 0';
            canvasContainer.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
        }, { passive: false });

        // Pan with Middle Click or Space + Left Click
        container.addEventListener('mousedown', (e) => {
            if (e.button === 1 || (e.button === 0 && this.isSpacePressed)) {
                e.preventDefault();
                this.isPanning = true;
                this.startPanX = e.clientX - this.panX;
                this.startPanY = e.clientY - this.panY;
                this.canvas.style.cursor = 'grabbing';
                return;
            }

            if (window.isInteractiveMode || window.isSamModeActive || window.isLamModeActive) return;

            if (this.annotator && this.annotator.onMouseDown) {
                const pt = this.getCanvasPoint(e);
                this.annotator.onMouseDown(pt, e);
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (this.isPanning) {
                this.panX += e.movementX;
                this.panY += e.movementY;
                const canvasContainer = document.getElementById('canvas-container');
                canvasContainer.style.transformOrigin = '0 0';
                canvasContainer.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
                return;
            }

            if (window.isInteractiveMode || window.isSamModeActive || window.isLamModeActive) return;

            if (this.annotator && this.annotator.onMouseMove) {
                const pt = this.getCanvasPoint(e);
                this.annotator.onMouseMove(pt, e);
            }
        });

        window.addEventListener('mouseup', (e) => {
            if (this.isPanning) {
                this.isPanning = false;
                this.canvas.style.cursor = this.isSpacePressed ? 'grab' : 'default';
                return;
            }

            if (window.isInteractiveMode || window.isSamModeActive || window.isLamModeActive) return;

            if (this.annotator && this.annotator.onMouseUp) {
                const pt = this.getCanvasPoint(e);
                this.annotator.onMouseUp(pt, e);
            }
        });

        // Context Menu (Right Click)
        container.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            if (this.annotator && this.annotator.onContextMenu) {
                const pt = this.getCanvasPoint(e);
                this.annotator.onContextMenu(pt, e);
            }
        });
    }

    getCanvasPoint(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    loadFrame(frameNumber, imageUrl) {
        this.currentFrame = frameNumber;
        this.fetchAnnotations();
    }

    fetchAnnotations() {
        fetch(`/getFrameAnnotations?video_uuid=${this.videoUuid}&frame_number=${this.currentFrame}`)
            .then(res => res.json())
            .then(data => {
                if (data.success && data.annotations) {
                    if (typeof data.annotations === 'string') {
                        try { this.annotations = JSON.parse(data.annotations); } catch(e) {}
                    } else if (typeof data.annotations === 'object') {
                        this.annotations = data.annotations;
                    }
                    if (!this.annotations.objects) this.annotations.objects = [];
                    if (!this.annotations.classifications) this.annotations.classifications = [];
                } else {
                    this.annotations = { objects: [], classifications: [] };
                }
                if (this.annotations && this.annotations.is_ambiguous) {
                    this.annotations.is_ambiguous = false;
                    const savePromise = this.saveAnnotations(false);
                    if (savePromise && typeof savePromise.then === 'function') {
                        savePromise.then(() => {
                            if (typeof window.updateAmbiguousCountBadge === 'function') {
                                window.updateAmbiguousCountBadge();
                            }
                        });
                    } else {
                        if (typeof window.updateAmbiguousCountBadge === 'function') {
                            window.updateAmbiguousCountBadge();
                        }
                    }
                    if (typeof window.showToast === 'function') {
                        window.showToast(`✅ 帧 #${this.currentFrame} 已完成消歧义复核`, 2000);
                    }
                }
                // Reset history stack for this frame
                this.history = [JSON.parse(JSON.stringify(this.annotations))];
                this.historyIndex = 0;

                this.updateSidebarList();
                this.render();
            })
            .catch(err => {
                console.error('[AnnotationCore] 加载标注数据失败:', err);
                // 网络异常时重置为空状态，避免界面卡死
                this.annotations = { objects: [], classifications: [] };
                this.history = [JSON.parse(JSON.stringify(this.annotations))];
                this.historyIndex = 0;
                this.updateSidebarList();
                this.render();
                if (typeof window.showToast === 'function') {
                    window.showToast('⚠️ 标注数据加载失败，请检查网络连接', 3000);
                }
            });
    }

    saveAnnotations(recordHistory = true) {
        if (recordHistory) {
            this.saveStateToHistory();
        }
        this.updateSidebarList();
        return fetch('/saveFrameAnnotations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_uuid: this.videoUuid,
                frame_number: this.currentFrame,
                annotations_json: JSON.stringify(this.annotations)
            })
        }).catch(err => {
            // 保存失败时明确通知用户，防止标注数据静默丢失
            console.error('[AnnotationCore] 标注保存失败:', err);
            if (typeof window.showToast === 'function') {
                window.showToast('❌ 标注保存失败！请检查网络或重新加载页面。', 5000);
            } else {
                alert('标注保存失败，请检查网络连接！');
            }
        });
    }

    updateSidebarList() {
        const countSpan = document.getElementById('segmentation-count');
        const listDiv = document.getElementById('segmentation-object-list');
        if (!listDiv) return;

        const objects = this.annotations.objects || [];
        if (countSpan) countSpan.textContent = objects.length;

        listDiv.innerHTML = '';
        objects.forEach((obj, idx) => {
            const isSelected = obj.id === this.selectedObjectId;
            const item = document.createElement('div');
            item.className = `d-flex justify-content-between align-items-center p-2 mb-1 border rounded ${isSelected ? 'bg-primary text-white border-primary' : 'bg-dark text-light border-secondary'}`;
            item.style.cursor = 'pointer';

            const name = document.createElement('span');
            name.className = 'font-weight-bold small';
            name.textContent = `#${idx + 1} [${obj.label || 'object'}]`;
            item.appendChild(name);

            const delBtn = document.createElement('button');
            delBtn.className = 'btn btn-sm btn-danger py-0 px-2';
            delBtn.innerHTML = '<i class="bi bi-trash"></i>';
            delBtn.onclick = (e) => {
                e.stopPropagation();
                this.annotations.objects.splice(idx, 1);
                if (this.selectedObjectId === obj.id) this.selectedObjectId = null;
                this.saveAnnotations();
                this.render();
            };
            item.appendChild(delBtn);

            item.onclick = () => {
                this.selectedObjectId = obj.id;
                this.updateSidebarList();
                this.render();
            };

            listDiv.appendChild(item);
        });
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.save();
        // NOTE: Viewport panning & zooming is handled by CSS transform on #canvas-container.
        // Do not apply ctx.translate / ctx.scale here to avoid double-transforming canvas drawings.

        // 2. Delegate Saved Shape Rendering to Active Annotator Plugin
        if (this.annotator && this.annotator.render) {
            this.annotator.render(this.ctx, this.annotations, this.selectedObjectId);
        }

        // 3. Draw Reviewable AI Suggestions (from database suggested_bboxes_text)
        if (typeof window.suggestionBboxes !== 'undefined' && window.suggestionBboxes.length > 0) {
            const thresholdInput = document.getElementById('suggestion-threshold');
            const threshold = thresholdInput ? parseFloat(thresholdInput.value) : 0.5;
            window.suggestionBboxes.forEach((sug) => {
                if (sug.score >= threshold) {
                    let polyPoints = sug.polygon;
                    if (!polyPoints || polyPoints.length < 3) {
                        const b = sug.box;
                        polyPoints = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
                    }
                    this.ctx.beginPath();
                    this.ctx.moveTo(polyPoints[0][0], polyPoints[0][1]);
                    for (let i = 1; i < polyPoints.length; i++) {
                        this.ctx.lineTo(polyPoints[i][0], polyPoints[i][1]);
                    }
                    this.ctx.closePath();
                    this.ctx.strokeStyle = 'rgba(46, 154, 255, 0.8)';
                    this.ctx.lineWidth = 1.5 / this.zoom;
                    this.ctx.fillStyle = 'rgba(46, 154, 255, 0.2)';
                    this.ctx.fill();
                    this.ctx.stroke();
                }
            });
        }

        // 4. Draw Interactive Preview Polygons (Smart Select / Find Similar Current Frame Results)
        if (typeof window.interactivePreviewBboxes !== 'undefined' && window.interactivePreviewBboxes.length > 0) {
            const thresholdInput = document.getElementById('result-threshold');
            const threshold = thresholdInput ? parseFloat(thresholdInput.value) : 0.5;
            window.interactivePreviewBboxes.forEach((res, index) => {
                if (res.score >= threshold) {
                    let polyPoints = res.polygon;
                    if (!polyPoints || polyPoints.length < 3) {
                        const b = res.box;
                        polyPoints = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
                    }
                    this.ctx.beginPath();
                    this.ctx.moveTo(polyPoints[0][0], polyPoints[0][1]);
                    for (let i = 1; i < polyPoints.length; i++) {
                        this.ctx.lineTo(polyPoints[i][0], polyPoints[i][1]);
                    }
                    this.ctx.closePath();

                    // Highlight brightly if mouse is hovering over this preview polygon
                    if (typeof window.hoveredPreviewIndex !== 'undefined' && window.hoveredPreviewIndex === index) {
                        this.ctx.strokeStyle = '#FFD000';
                        this.ctx.lineWidth = 3 / this.zoom;
                        this.ctx.fillStyle = 'rgba(255, 208, 0, 0.4)';
                    } else {
                        this.ctx.strokeStyle = 'rgba(255, 171, 0, 0.8)';
                        this.ctx.lineWidth = 2 / this.zoom;
                        this.ctx.fillStyle = 'rgba(255, 171, 0, 0.15)';
                    }

                    this.ctx.fill();
                    this.ctx.stroke();
                }
            });
        }

        // 5. Draw Interactive Prompt BBoxes (Smart Select Drag-Prompt Inputs)
        if (window.isInteractiveMode) {
            if (typeof window.positiveExampleBboxes !== 'undefined' && window.positiveExampleBboxes.length > 0) {
                this.ctx.strokeStyle = 'rgba(94, 148, 117, 0.9)';
                this.ctx.lineWidth = 2 / this.zoom;
                this.ctx.setLineDash([5 / this.zoom, 3 / this.zoom]);
                window.positiveExampleBboxes.forEach(box => {
                    if (box) {
                        const x1 = box.x1 !== undefined ? box.x1 : box[0];
                        const y1 = box.y1 !== undefined ? box.y1 : box[1];
                        const x2 = box.x2 !== undefined ? box.x2 : box[2];
                        const y2 = box.y2 !== undefined ? box.y2 : box[3];
                        this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                    }
                });
                this.ctx.setLineDash([]);
            }
            if (typeof window.negativeExampleBboxes !== 'undefined' && window.negativeExampleBboxes.length > 0) {
                this.ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
                this.ctx.lineWidth = 2 / this.zoom;
                this.ctx.setLineDash([5 / this.zoom, 3 / this.zoom]);
                window.negativeExampleBboxes.forEach(box => {
                    if (box) {
                        const x1 = box.x1 !== undefined ? box.x1 : box[0];
                        const y1 = box.y1 !== undefined ? box.y1 : box[1];
                        const x2 = box.x2 !== undefined ? box.x2 : box[2];
                        const y2 = box.y2 !== undefined ? box.y2 : box[3];
                        this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                    }
                });
                this.ctx.setLineDash([]);
            }
        }

        // 6. Draw Temporary LAM Suggestion Polygon
        if (typeof window.tempLamPolygon !== 'undefined' && window.tempLamPolygon) {
            const polyPoints = window.tempLamPolygon;
            this.ctx.beginPath();
            this.ctx.moveTo(polyPoints[0][0], polyPoints[0][1]);
            for (let i = 1; i < polyPoints.length; i++) {
                this.ctx.lineTo(polyPoints[i][0], polyPoints[i][1]);
            }
            this.ctx.closePath();
            this.ctx.strokeStyle = 'rgba(255, 171, 0, 0.9)';
            this.ctx.lineWidth = 3 / this.zoom;
            this.ctx.fillStyle = 'rgba(255, 171, 0, 0.25)';
            this.ctx.fill();
            this.ctx.stroke();
        }

        this.ctx.restore();
    }
}