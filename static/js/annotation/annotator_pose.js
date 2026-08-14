/**
 * PoseAnnotator - Manual & AI Keypoint / Skeleton Annotator Plugin
 *
 * 设计目标：完全不写死点位数量或名称。每个"骨架实例"就是一个
 * AnnotationObject { type:'keypoint', label, bbox, keypoints:[{name,x,y,v}, ...] }，
 * keypoints 数组长度、顺序、命名完全自由 —— 可以是 COCO-17，也可以是 6 点的机械臂，
 * 21 点的手部，或者任何你想标的东西。
 *
 * 集成 SAM 2.1 实例隔离 + GKDT 姿态生成交互引擎，彻底消除跨目标混淆。
 */
class PoseAnnotator {
    constructor(core) {
        this.core = core;

        this.selectedKeypointIndex = null;
        this.draggedKeypointIndex = null;
        this.isDraggingPoint = false;

        this.isPlacingNewInstance = false;
        this.armedClass = null;

        this.isFreeformAdding = false;
        this.activeNewInstance = null;

        this.isGkdtModeActive = false;

        this.hoverPoint = null;

        this.nameInputEl = null;

        this.initDOM();
        this.bindShortcuts();
        this.updateSidebarList();
    }

    // ============================================================
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

        if (window.classKeypointSchemas && window.classKeypointSchemas[label]) {
            const s = typeof window.classKeypointSchemas[label] === 'string'
                ? JSON.parse(window.classKeypointSchemas[label])
                : window.classKeypointSchemas[label];
            if (s && Array.isArray(s.points) && s.points.length > 0) {
                return { points: s.points, edges: Array.isArray(s.edges) ? s.edges : [] };
            }
        }

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

        $.ajax({
            url: '/api/saveClassKeypointSchema',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ label: label, schema: schemaData }),
            success: function(res) {
                if (!res.success) {
                    console.warn("Failed to persist keypoint schema to database:", res.message);
                }
            },
            error: function(xhr, status, err) {
                console.warn("Failed to persist keypoint schema to database:", err);
            }
        });
    }

    // ============================================================
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
        return '#e4e4e7';
    }

    // ============================================================
    // ============================================================

    initDOM() {
        const self = this;

        $(document).off('click', '#btn-show-gkdt-panel').on('click', '#btn-show-gkdt-panel', function () {
            $('#btn-show-gkdt-panel').removeClass('btn-outline-secondary').addClass('btn-outline-danger active');
            $('#btn-show-truelam-panel').removeClass('btn-outline-warning active').addClass('btn-outline-secondary');
            $('#btn-show-interp-panel').removeClass('btn-outline-info active').addClass('btn-outline-secondary');
            $('#gkdt-panel-controls').show();
            $('#truelam-panel-controls, #interp-panel-controls').hide();
        });

        $(document).off('click', '#btn-show-truelam-panel').on('click', '#btn-show-truelam-panel', function () {
            $('#btn-show-truelam-panel').removeClass('btn-outline-secondary').addClass('btn-outline-warning active');
            $('#btn-show-gkdt-panel').removeClass('btn-outline-danger active').addClass('btn-outline-secondary');
            $('#btn-show-interp-panel').removeClass('btn-outline-info active').addClass('btn-outline-secondary');
            $('#truelam-panel-controls').show();
            $('#gkdt-panel-controls, #interp-panel-controls').hide();
        });

        $(document).off('click', '#btn-show-interp-panel').on('click', '#btn-show-interp-panel', function () {
            $('#btn-show-interp-panel').removeClass('btn-outline-secondary').addClass('btn-outline-info active');
            $('#btn-show-gkdt-panel').removeClass('btn-outline-danger active').addClass('btn-outline-secondary');
            $('#btn-show-truelam-panel').removeClass('btn-outline-warning active').addClass('btn-outline-secondary');
            $('#interp-panel-controls').show();
            $('#gkdt-panel-controls, #truelam-panel-controls').hide();
        });

        $(document).off('click', '#btn-toggle-gkdt-interactive').on('click', '#btn-toggle-gkdt-interactive', function () {
            self.isGkdtModeActive = !self.isGkdtModeActive;
            const $btn = $(this);

            if (self.isGkdtModeActive) {
                $btn.removeClass('btn-outline-danger').addClass('btn-danger text-white')
                    .html('<i class="bi bi-crosshair mr-1"></i>GKDT Point-Click ON (Click Target)');
                self.core.canvas.style.cursor = 'crosshair';
                if (typeof window.showToast === 'function') {
                    window.showToast('🎯 GKDT Point-Click Active: Click any target object in image, SAM 2.1 + GKDT will extract its pose!', 3500);
                }
            } else {
                $btn.removeClass('btn-danger text-white').addClass('btn-outline-danger')
                    .html('<i class="bi bi-crosshair mr-1"></i>Enable GKDT Point-Click Mode');
                self.core.canvas.style.cursor = 'default';
                if (typeof window.showToast === 'function') {
                    window.showToast('⏸️ Switched to manual pose annotation mode', 2000);
                }
            }
        });

        $(document).off('click', '#btn-truelam-pose-batch').on('click', '#btn-truelam-pose-batch', function (e) {
            if (e) { e.preventDefault(); e.stopPropagation(); }

            let cls = self.getSelectedClass();
            const customPrompt = $('#pose-truelam-prompt-input').val().trim();

            if (!cls) {
                if (customPrompt) {
                    cls = customPrompt;
                } else {
                    const firstClassLi = $('#class-list li').first();
                    if (firstClassLi.length && firstClassLi.data('class-name')) {
                        cls = firstClassLi.data('class-name');
                    }
                }
            }

            if (!cls) {
                const warnMsg = '⚠️ Please select a class or enter target text (e.g. person, dog)';
                if (typeof window.showToast === 'function') {
                    window.showToast(warnMsg, 3500);
                } else {
                    alert(warnMsg);
                }
                return;
            }

            const promptToUse = customPrompt || cls;
            const $btn = $(this);
            $btn.prop('disabled', true).html('<span class="spinner-border spinner-border-sm mr-1"></span> SAM3 + GKDT Detecting...');

            if (typeof window.showToast === 'function') {
                window.showToast(`🔍 Running SAM3 open-vocabulary detection [${promptToUse}] -> [${cls}] pose generation...`, 4000);
            }

            const currentFrame = (typeof self.core.currentFrame !== 'undefined') ? self.core.currentFrame : parseInt($('#frame-slider').val() || '0', 10);
            const videoUuid = self.core.videoUuid || (window.annotationCore ? window.annotationCore.videoUuid : '');

            $.ajax({
                url: '/api/gkdt_sam3_batch_pose_predict',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    video_uuid: videoUuid,
                    frame_number: currentFrame,
                    class_label: cls,
                    text_prompt: promptToUse,
                    confidence: 0.25
                }),
                success: function(res) {
                    $btn.prop('disabled', false).html('<i class="bi bi-magic mr-1"></i>Detect Poses on Current Frame');
                    if (res.success && res.pose_objects && res.pose_objects.length > 0) {
                        if (typeof window.showToast === 'function') {
                            window.showToast(`🎉 Successfully detected & attached ${res.pose_objects.length} pose skeletons for '${promptToUse}'!`, 3500);
                        }

                        if (!self.core.annotations) self.core.annotations = { objects: [] };
                        if (!self.core.annotations.objects) self.core.annotations.objects = [];

                        res.pose_objects.forEach(obj => {
                            self.core.annotations.objects.push(obj);
                        });

                        if (typeof self.core.saveAnnotations === 'function') {
                            self.core.saveAnnotations();
                        }
                        if (typeof self.updateSidebarList === 'function') {
                            self.updateSidebarList();
                        }
                        if (typeof self.core.render === 'function') {
                            self.core.render();
                        }
                    } else {
                        const warnMsg = `⚠️ SAM3 could not detect targets matching '${promptToUse}' in current frame`;
                        if (typeof window.showToast === 'function') {
                            window.showToast(warnMsg, 3500);
                        } else {
                            alert(warnMsg);
                        }
                    }
                },
                error: function(err) {
                    $btn.prop('disabled', false).html('<i class="bi bi-magic mr-1"></i>Detect Poses on Current Frame');
                    const msg = (err.responseJSON && err.responseJSON.message) ? err.responseJSON.message : 'Pose detection service error';
                    if (typeof window.showToast === 'function') {
                        window.showToast('⚠️ ' + msg, 3500);
                    } else {
                        alert('⚠️ ' + msg);
                    }
                }
            });
        });

        $(document).off('click', '#btn-truelam-pose-dataset-batch').on('click', '#btn-truelam-pose-dataset-batch', function (e) {
            if (e) { e.preventDefault(); e.stopPropagation(); }

            let cls = self.getSelectedClass();
            const customPrompt = $('#pose-truelam-prompt-input').val().trim();

            if (!cls) {
                if (customPrompt) {
                    cls = customPrompt;
                } else {
                    const firstClassLi = $('#class-list li').first();
                    if (firstClassLi.length && firstClassLi.data('class-name')) {
                        cls = firstClassLi.data('class-name');
                    }
                }
            }

            if (!cls) {
                const warnMsg = '⚠️ Please select a class or enter target text (e.g. person, dog)';
                if (typeof window.showToast === 'function') window.showToast(warnMsg, 3500);
                else alert(warnMsg);
                return;
            }

            const promptToUse = customPrompt || cls;
            if (!confirm(`🚀 Start SAM3 + GKDT Pose Auto-Labeling for full dataset?\n\nTarget Class: [${cls}]\nSearch Prompt: [${promptToUse}]\n\nPose skeletons will be automatically generated for all video frames in the background.`)) {
                return;
            }

            const $btn = $(this);
            $btn.prop('disabled', true).html('<span class="spinner-border spinner-border-sm mr-1"></span> Processing Dataset...');

            let videoUuids = [self.core.videoUuid || (window.annotationCore ? window.annotationCore.videoUuid : '')];
            if (window.datasetVideoUuids && Array.isArray(window.datasetVideoUuids) && window.datasetVideoUuids.length > 0) {
                videoUuids = window.datasetVideoUuids;
            }

            $.ajax({
                url: '/api/apply_pose_class_to_videos',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    video_uuids: videoUuids,
                    class_name: cls,
                    confidence_threshold: 0.25,
                    process_all_frames: true
                }),
                success: function(res) {
                    $btn.prop('disabled', false).html('<i class="bi bi-collection-play mr-1"></i>Propagate Poses to Dataset (All Videos)');
                    if (res.success && res.task_uuid) {
                        if (typeof window.showToast === 'function') {
                            window.showToast(`🚀 Dataset pose auto-labeling task started in background! (Class: ${cls})`, 4000);
                        } else {
                            alert(`🚀 Dataset pose auto-labeling task started in background!`);
                        }
                    } else {
                        const warnMsg = '⚠️ Failed to start dataset task: ' + (res.message || 'Unknown error');
                        if (typeof window.showToast === 'function') window.showToast(warnMsg, 3500);
                        else alert(warnMsg);
                    }
                },
                error: function(err) {
                    $btn.prop('disabled', false).html('<i class="bi bi-collection-play mr-1"></i>Propagate Poses to Dataset (All Videos)');
                    const msg = (err.responseJSON && err.responseJSON.message) ? err.responseJSON.message : 'Task launch failed';
                    if (typeof window.showToast === 'function') window.showToast('⚠️ ' + msg, 3500);
                    else alert('⚠️ ' + msg);
                }
            });
        });

        $(document).off('click', '#btn-set-pose-interp-start').on('click', '#btn-set-pose-interp-start', function () {
            const currFrame = self.core.currentFrame;
            $('#pose-interp-start-frame').val(currFrame);
            if (typeof window.showToast === 'function') {
                window.showToast(`📍 Set frame #${currFrame} as interpolation start frame`, 2000);
            }
        });

        $(document).off('click', '#btn-set-pose-interp-end').on('click', '#btn-set-pose-interp-end', function () {
            const currFrame = self.core.currentFrame;
            $('#pose-interp-end-frame').val(currFrame);
            if (typeof window.showToast === 'function') {
                window.showToast(`📍 Set frame #${currFrame} as interpolation end frame`, 2000);
            }
        });

        $(document).off('click', '#btn-exec-pose-interpolate').on('click', '#btn-exec-pose-interpolate', function () {
            const startFrame = parseInt($('#pose-interp-start-frame').val());
            const endFrame = parseInt($('#pose-interp-end-frame').val());

            if (isNaN(startFrame) || isNaN(endFrame)) {
                const msg = '⚠️ Please specify both Start Frame and End Frame for interpolation!';
                if (typeof window.showToast === 'function') window.showToast(msg, 3000);
                else alert(msg);
                return;
            }

            if (startFrame >= endFrame) {
                const msg = '⚠️ End frame must be greater than start frame!';
                if (typeof window.showToast === 'function') window.showToast(msg, 3000);
                else alert(msg);
                return;
            }

            const selectedObjId = self.core.selectedObjectId;
            const $btn = $(this);
            $btn.prop('disabled', true).html('<i class="spinner-border spinner-border-sm mr-1"></i>Interpolating...');

            $.ajax({
                url: '/api/interpolatePoseKeypoints',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    video_uuid: self.core.videoUuid,
                    object_id: selectedObjId || null,
                    start_frame_number: startFrame,
                    end_frame_number: endFrame
                }),
                success: function (res) {
                    $btn.prop('disabled', false).html('<i class="bi bi-arrow-down-up mr-1"></i>Interpolate Keypoints');
                    if (res.success) {
                        if (typeof window.showToast === 'function') {
                            window.showToast(`✨ ${res.message}`, 4000);
                        } else {
                            alert(res.message);
                        }
                        self.core.fetchAnnotations();
                    } else {
                        const err = res.message || 'Keypoint interpolation failed';
                        if (typeof window.showToast === 'function') window.showToast(`❌ ${err}`, 3500);
                        else alert(err);
                    }
                },
                error: function (xhr) {
                    $btn.prop('disabled', false).html('<i class="bi bi-arrow-down-up mr-1"></i>Interpolate Keypoints');
                    const err = xhr.responseJSON?.message || 'Server timeout';
                    if (typeof window.showToast === 'function') window.showToast(`❌ ${err}`, 3500);
                    else alert(err);
                }
            });
        });

        $(document).off('click', '#btn-add-skeleton').on('click', '#btn-add-skeleton', function () {
            const cls = self.getSelectedClass();
            if (!cls) {
                if (typeof window.showToast === 'function') {
                    window.showToast('⚠️ Please select/create a category in the right Class Registry first', 3000);
                }
                return;
            }
            self.isPlacingNewInstance = true;
            self.armedClass = cls;
            self.isFreeformAdding = false;
            if (typeof window.showToast === 'function') {
                window.showToast(`🎯 Click anywhere on canvas to place [${cls}] skeleton`, 2500);
            }
        });

        $(document).off('click', '#btn-edit-pose-schema').on('click', '#btn-edit-pose-schema', function () {
            const cls = self.getSelectedClass();
            if (!cls) {
                if (typeof window.showToast === 'function') {
                    window.showToast('⚠️ Please select/create a category in the right Class Registry first', 3000);
                }
                return;
            }
            self.openSchemaEditor(cls);
        });

        $(document).off('click', '#btn-finish-pose-freeform').on('click', '#btn-finish-pose-freeform', function () {
            self.finishFreeformAdding();
        });

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
                    if (typeof window.showToast === 'function') window.showToast('Placement cancelled', 1200);
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

            if (['0', '1', '2'].includes(e.key) && self.selectedKeypointIndex !== null) {
                e.preventDefault();
                self.setSelectedKeypointVisibility(parseInt(e.key));
            }
        });
    }

    // ============================================================
    // ============================================================

    showNameInput(screenX, screenY, defaultName) {
        this.cancelNameInput();
        const input = document.createElement('input');
        input.type = 'text';
        input.value = defaultName || '';
        input.placeholder = 'Keypoint name, e.g. nose / joint_1 ...';
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
        if (typeof window.showToast === 'function') window.showToast('✅ Skeleton annotation complete', 1500);
    }

    // ============================================================
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
            obj.keypoints.splice(this.selectedKeypointIndex, 1);
            this.selectedKeypointIndex = null;
            this.recomputeBbox(obj);
            this.core.saveAnnotations();
            this.updateVisibilityButtonsUI();
            this.updateSidebarList();
            this.core.render();
        } else if (this.core.selectedObjectId) {
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
                window.showToast('📍 Placed according to template, drag points to correct position', 2500);
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
                window.showToast('✏️ No template set: Click canvas to add keypoints, Enter/Right-click/"Done" button to finish', 3500);
            }
        }
    }

    // ============================================================
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

        if (this.nameInputEl) return;

        if (this.isGkdtModeActive) {
            const cls = this.getSelectedClass();
            if (!cls) {
                if (typeof window.showToast === 'function') window.showToast('⚠️ Please select/create a category in the right Class Registry first!', 3000);
                return;
            }

            const currentFrame = this.core.currentFrame || parseInt($('#frame-slider').val() || '0', 10);
            const self = this;

            if (typeof window.showToast === 'function') {
                window.showToast('⚡ SAM 2.1 isolating target -> GKDT generating independent pose...', 4000);
            }

            $.ajax({
                url: '/api/gkdt_sam_pose_predict',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({
                    video_uuid: self.core.videoUuid,
                    frame_number: currentFrame,
                    class_label: cls,
                    point: { x: Math.round(pt.x), y: Math.round(pt.y) }
                }),
                success: function(res) {
                    if (res.success && res.pose_object) {
                        self.core.annotations.objects.push(res.pose_object);
                        self.core.selectedObjectId = res.pose_object.id;
                        self.selectedKeypointIndex = null;

                        self.core.saveAnnotations();
                        self.updateSidebarList();
                        self.core.render();
                        if (typeof window.showToast === 'function') {
                            window.showToast(`✨ SAM 2.1 target isolation complete! Generated [${cls}] pose`, 2500);
                        }
                    } else {
                        if (typeof window.showToast === 'function') window.showToast('⚠️ Recognition failed: ' + (res.message || 'Unknown error'), 3500);
                    }
                },
                error: function(xhr) {
                    const msg = xhr.responseJSON ? xhr.responseJSON.message : 'Server communication failed';
                    if (typeof window.showToast === 'function') window.showToast('⚠️ Error: ' + msg, 3500);
                }
            });
            return;
        }

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
    // ============================================================

    colorForVisibility(v) {
        if (v === 0) return 'rgba(160,160,170,0.9)';   // absent - Gray
        if (v === 1) return '#b59656';                  // occluded - Soft yellow/orange
        return '#5e9475';                                // visible - Sage green
    }

    render(ctx, annotations, selectedId) {
        const objects = (annotations.objects || []).filter(o => o.type === 'keypoint');
        const zoom = this.core.zoom;

        objects.forEach(obj => {
            const isSelectedInstance = obj.id === selectedId;
            const classColor = this.getColorForClass(obj.label);

            if (obj.bbox && obj.bbox.length === 4) {
                const bx1 = parseFloat(obj.bbox[0]);
                const by1 = parseFloat(obj.bbox[1]);
                const bx2 = parseFloat(obj.bbox[2]);
                const by2 = parseFloat(obj.bbox[3]);

                if (!isNaN(bx1) && !isNaN(by1) && !isNaN(bx2) && !isNaN(by2)) {
                    ctx.save();
                    ctx.strokeStyle = classColor;
                    ctx.lineWidth = (isSelectedInstance ? 2.0 : 1.0) / zoom;
                    ctx.setLineDash([4 / zoom, 4 / zoom]);
                    ctx.globalAlpha = isSelectedInstance ? 0.9 : 0.45;
                    ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);
                    ctx.restore();
                }
            }

            const kps = (obj.keypoints || []).map(k => ({
                name: k.name || '',
                x: parseFloat(k.x) || 0,
                y: parseFloat(k.y) || 0,
                v: (typeof k.v !== 'undefined') ? parseInt(k.v) : 2
            }));

            const byName = {};
            kps.forEach(k => { if (k.name) byName[k.name] = k; });

            const schema = this.getSchema(obj.label);
            let drewAnyEdge = false;
            if (schema && schema.edges && schema.edges.length > 0) {
                ctx.save();
                ctx.strokeStyle = classColor;
                ctx.lineWidth = (isSelectedInstance ? 2.5 : 1.5) / zoom;
                ctx.globalAlpha = isSelectedInstance ? 0.95 : 0.65;
                schema.edges.forEach(([a, b]) => {
                    const pa = byName[a], pb = byName[b];
                    if (pa && pb && pa.v > 0 && pb.v > 0) {
                        ctx.beginPath();
                        ctx.moveTo(pa.x, pa.y);
                        ctx.lineTo(pb.x, pb.y);
                        ctx.stroke();
                        drewAnyEdge = true;
                    }
                });
                ctx.restore();
            }

            if (!drewAnyEdge && kps.length > 1) {
                ctx.save();
                ctx.strokeStyle = classColor;
                ctx.lineWidth = (isSelectedInstance ? 2.0 : 1.2) / zoom;
                ctx.globalAlpha = 0.65;
                const visKps = kps.filter(k => k.v > 0);
                for (let i = 0; i < visKps.length - 1; i++) {
                    ctx.beginPath();
                    ctx.moveTo(visKps[i].x, visKps[i].y);
                    ctx.lineTo(visKps[i + 1].x, visKps[i + 1].y);
                    ctx.stroke();
                }
                ctx.restore();
            }

            kps.forEach((kp, idx) => {
                const v = kp.v;
                const isSelectedPoint = isSelectedInstance && idx === this.selectedKeypointIndex;
                const radius = (isSelectedPoint ? 7 : 5) / zoom;

                ctx.save();
                ctx.beginPath();
                ctx.arc(kp.x, kp.y, radius, 0, Math.PI * 2);
                ctx.fillStyle = this.colorForVisibility(v);
                ctx.globalAlpha = v === 0 ? 0.45 : (isSelectedInstance ? 1.0 : 0.85);
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

        if ((this.isPlacingNewInstance || this.isFreeformAdding || this.isGkdtModeActive) && this.hoverPoint) {
            ctx.save();
            ctx.strokeStyle = this.isGkdtModeActive ? 'rgba(189, 99, 99, 0.9)' : 'rgba(244, 244, 245, 0.9)';
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
            wrapper.className = `mb-2 p-2 border rounded ${isSelected ? 'border-primary' : ''}`;
            wrapper.style.background = isSelected ? 'var(--bg-surface-secondary)' : 'var(--bg-surface)';
            wrapper.style.borderColor = isSelected ? 'var(--color-primary-accent)' : 'var(--border-color)';

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
    // ============================================================

    openSchemaEditor(label) {
        $('#pose-schema-modal').remove();

        const schema = this.getSchema(label);
        const self = this;

        const modal = $(`
            <div id="pose-schema-modal" class="keybind-modal-backdrop">
                <div class="keybind-modal-card" style="width: 480px; text-align: left; max-height: 82vh; overflow-y: auto;">
                    <div class="keybind-modal-header">
                        <span>✏️ Point Template — <span class="target-class-pill" style="font-size:0.8rem;">${label}</span></span>
                        <button type="button" class="close text-light ml-auto" id="pose-schema-close-x">&times;</button>
                    </div>
                    <div class="small text-muted mb-2">
                        This template only affects which points are automatically placed when clicking "Add Skeleton", not the only format restriction—
                        You can manually add/delete points at any time during annotation, the number is completely unlimited.
                    </div>

                    <div class="d-flex justify-content-between align-items-center mt-3 mb-1">
                        <b class="small text-uppercase text-info">Point List (Points)</b>
                        <button class="btn btn-xs btn-outline-success" id="pose-schema-load-coco17">Load COCO-17 example template</button>
                    </div>
                    <div id="pose-schema-points-list"></div>
                    <div class="input-group input-group-sm mt-2">
                        <input type="text" id="pose-schema-new-point" class="form-control tool-input" placeholder="New point name, e.g. head / hinge_a ...">
                        <div class="input-group-append">
                            <button class="btn btn-sm btn-outline-info" id="pose-schema-add-point">+ Add</button>
                        </div>
                    </div>

                    <div class="mt-3 mb-1"><b class="small text-uppercase text-info">Skeleton Connections (Bones, Optional)</b></div>
                    <div id="pose-schema-edges-list"></div>
                    <div class="input-group input-group-sm mt-2">
                        <select id="pose-schema-edge-a" class="form-control tool-input"></select>
                        <select id="pose-schema-edge-b" class="form-control tool-input"></select>
                        <div class="input-group-append">
                            <button class="btn btn-sm btn-outline-info" id="pose-schema-add-edge">+ Connect</button>
                        </div>
                    </div>

                    <div class="mt-3 mb-1"><b class="small text-uppercase text-info">Import / Export JSON</b></div>
                    <textarea id="pose-schema-json" class="form-control tool-input" rows="4" style="font-family: monospace; font-size: 0.75rem;"></textarea>
                    <div class="mt-1">
                        <button class="btn btn-xs btn-outline-secondary" id="pose-schema-apply-json">Apply JSON above</button>
                    </div>

                    <div class="keybind-modal-footer">
                        <button class="btn btn-sm btn-secondary mr-2" id="pose-schema-cancel">Close</button>
                        <button class="btn btn-sm btn-primary" id="pose-schema-save">Save Template</button>
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
                list.append('<div class="text-muted small">No points, manually add below, or load example template.</div>');
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
                list.append('<div class="text-muted small">No connections (connections only affect display, not annotation data).</div>');
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
                if (typeof window.showToast === 'function') window.showToast('⚠️ Point with same name already exists', 2000);
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
            if (typeof window.showToast === 'function') window.showToast('COCO-17 example loaded (not saved yet, can continue editing)', 2000);
        });

        $('#pose-schema-apply-json').on('click', function () {
            try {
                const parsed = JSON.parse($('#pose-schema-json').val());
                if (!Array.isArray(parsed.points)) throw new Error('points must be an array');
                working = { points: parsed.points, edges: Array.isArray(parsed.edges) ? parsed.edges : [] };
                renderPoints();
                renderEdgeSelectors();
                renderEdges();
                if (typeof window.showToast === 'function') window.showToast('JSON applied (not saved yet)', 1800);
            } catch (e) {
                if (typeof window.showToast === 'function') window.showToast('⚠️ JSON format error: ' + e.message, 3000);
            }
        });

        $('#pose-schema-save').on('click', () => {
            self.saveSchema(label, working);
            if (typeof window.showToast === 'function') window.showToast(`✅ [${label}] template saved (${working.points.length} points)`, 2000);
            modal.remove();
            self.core.render();
        });
        $('#pose-schema-cancel, #pose-schema-close-x').on('click', () => modal.remove());
    }
}