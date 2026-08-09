/**
 * SegmentationAnnotator - Professional Polygon Creation & Vertex Editing Plugin
 */
class SegmentationAnnotator {
    constructor(core) {
        this.core = core;
        this.currentPolygonPoints = [];
        this.hoverPoint = null;
        this.hoveredVertexIndex = null;
        this.draggedVertexIndex = null;
        this.hoveredPolygonId = null;

        this.activeTool = 'manual'; // 'manual', 'brush', 'eraser'
        this.brushRadius = 15;
        this.isPainting = false;
        this.brushStrokePoints = [];

        // DOM listeners for Paint Tools
        $(document).on('click', '#btn-brush-tool', (e) => {
            e.stopPropagation();
            this.activeTool = (this.activeTool === 'brush') ? 'manual' : 'brush';
            this.updateToolUI();
            this.core.render();
        });
        $(document).on('click', '#btn-eraser-tool', (e) => {
            e.stopPropagation();
            this.activeTool = (this.activeTool === 'eraser') ? 'manual' : 'eraser';
            this.updateToolUI();
            this.core.render();
        });
        $(document).on('input', '#brush-size-slider', (e) => {
            this.brushRadius = parseInt($(e.target).val());
            $('#brush-size-val').text(this.brushRadius + 'px');
            this.core.render();
        });

        // Key listeners for closing, canceling, or deleting polygon
        window.addEventListener('keydown', (e) => {
            if ($(e.target).is('input, textarea')) return;

            if (e.key === 'Enter') {
                this.closePolygon();
            } else if (e.key === 'Escape') {
                if (this.isPainting) {
                    this.isPainting = false;
                    this.brushStrokePoints = [];
                    this.core.render();
                } else if (this.currentPolygonPoints.length > 0) {
                    this.currentPolygonPoints = [];
                    this.core.render();
                } else if (this.core.selectedObjectId) {
                    this.core.selectedObjectId = null;
                    this.core.updateSidebarList();
                    this.core.render();
                }
            } else if (e.key === 'Delete' || e.key === 'Backspace') {
                if (this.core.selectedObjectId) {
                    const idx = this.core.annotations.objects.findIndex(o => o.id === this.core.selectedObjectId);
                    if (idx >= 0) {
                        this.core.annotations.objects.splice(idx, 1);
                        this.core.selectedObjectId = null;
                        this.core.saveAnnotations();
                        this.core.render();
                    }
                }
            }
        });
    }

    updateToolUI() {
        $('#btn-brush-tool').toggleClass('active btn-info text-white', this.activeTool === 'brush');
        $('#btn-eraser-tool').toggleClass('active btn-danger text-white', this.activeTool === 'eraser');
        if (this.activeTool === 'brush' || this.activeTool === 'eraser') {
            $('#brush-size-controls').slideDown(150);
        } else {
            $('#brush-size-controls').slideUp(150);
        }
    }

    getSelectedClass() {
        if (this.core.selectedClass) return this.core.selectedClass;
        if (typeof window.activeClass !== 'undefined' && window.activeClass) return window.activeClass;
        const activeLi = $('#class-list li.active');
        if (activeLi.length) {
            const cls = activeLi.data('class-name');
            if (cls) {
                this.core.selectedClass = cls;
                return cls;
            }
        }
        return null;
    }

    onMouseDown(pt, e) {
        // If middle mouse or space panning, ignore
        if (e.button === 1 || this.core.isSpacePressed) return;

        // Active Paint Brush / Eraser Tool
        if (this.activeTool === 'brush' || this.activeTool === 'eraser') {
            const currentClass = this.getSelectedClass();
            if (!currentClass && this.activeTool === 'brush') {
                if (typeof window.showToast === 'function') {
                    window.showToast("⚠️ Please select a Class from the right sidebar first!", 3000);
                }
                return;
            }
            this.isPainting = true;
            this.brushStrokePoints = [[pt.x, pt.y]];
            this.core.render();
            return;
        }

        // Double Click to finish current polygon
        if (e.detail === 2) {
            this.closePolygon();
            return;
        }

        // If currently drawing a polygon:
        if (this.currentPolygonPoints.length > 0) {
            // Check if clicking near the starting vertex to close
            const startPt = this.currentPolygonPoints[0];
            const distToStart = Math.hypot(pt.x - startPt[0], pt.y - startPt[1]);
            const snapRadius = 12 / this.core.zoom;

            if (this.currentPolygonPoints.length >= 3 && distToStart <= snapRadius) {
                this.closePolygon();
                return;
            }

            // Append point
            this.currentPolygonPoints.push([pt.x, pt.y]);
            this.core.render();
            return;
        }

        // If not drawing, check if clicking a vertex on the selected polygon to drag:
        if (this.core.selectedObjectId) {
            const selectedObj = this.core.annotations.objects.find(o => o.id === this.core.selectedObjectId);
            if (selectedObj && selectedObj.polygon) {
                const hitRadius = 10 / this.core.zoom;
                for (let i = 0; i < selectedObj.polygon.length; i++) {
                    const v = selectedObj.polygon[i];
                    if (Math.hypot(pt.x - v[0], pt.y - v[1]) <= hitRadius) {
                        this.draggedVertexIndex = i;
                        return;
                    }
                }
            }
        }

        // Check if clicking inside or near an existing polygon to select it:
        const objects = this.core.annotations.objects || [];
        for (let obj of objects) {
            if (obj.type === 'polygon' && obj.polygon) {
                if (this.isPointInPolygon([pt.x, pt.y], obj.polygon)) {
                    this.core.selectedObjectId = obj.id;
                    this.core.updateSidebarList();
                    this.core.render();
                    return;
                }
            }
        }

        // Otherwise, start a new polygon! First verify a class is selected:
        const currentClass = this.getSelectedClass();
        if (!currentClass) {
            if (typeof window.showToast === 'function') {
                window.showToast("⚠️ Please select a Class from the right sidebar first!", 3000);
            } else {
                alert("Please select a Class from the right sidebar first!");
            }
            return;
        }

        // Deselect any previous object and start new polygon
        this.core.selectedObjectId = null;
        this.core.updateSidebarList();
        this.currentPolygonPoints = [[pt.x, pt.y]];
        this.core.render();
    }

    onMouseMove(pt, e) {
        this.hoverPoint = pt;

        if (this.isPainting) {
            this.brushStrokePoints.push([pt.x, pt.y]);
            this.core.render();
            return;
        }

        // Handle vertex dragging
        if (this.draggedVertexIndex !== null && this.core.selectedObjectId) {
            const selectedObj = this.core.annotations.objects.find(o => o.id === this.core.selectedObjectId);
            if (selectedObj && selectedObj.polygon) {
                selectedObj.polygon[this.draggedVertexIndex] = [pt.x, pt.y];
                this.core.render();
                return;
            }
        }

        // Handle hovering state
        if (this.currentPolygonPoints.length > 0) {
            this.core.render();
        }
    }

    onMouseUp(pt, e) {
        if (this.isPainting) {
            this.isPainting = false;
            this.finishBrushStroke();
            this.brushStrokePoints = [];
            this.core.saveAnnotations();
            this.core.render();
            return;
        }

        if (this.draggedVertexIndex !== null) {
            this.draggedVertexIndex = null;
            this.core.saveAnnotations();
            this.core.render();
        }
    }

    finishBrushStroke() {
        if (!this.brushStrokePoints || this.brushStrokePoints.length === 0) return;

        const currentClass = this.getSelectedClass() || 'object';
        const r = this.brushRadius;
        let boundary = [];

        if (this.brushStrokePoints.length === 1) {
            const center = this.brushStrokePoints[0];
            const steps = 16;
            for (let i = 0; i < steps; i++) {
                const angle = (i / steps) * Math.PI * 2;
                boundary.push([center[0] + r * Math.cos(angle), center[1] + r * Math.sin(angle)]);
            }
        } else {
            const leftPts = [];
            const rightPts = [];
            for (let i = 0; i < this.brushStrokePoints.length; i++) {
                const p = this.brushStrokePoints[i];
                let dx = 0, dy = 0;
                if (i < this.brushStrokePoints.length - 1) {
                    dx = this.brushStrokePoints[i + 1][0] - p[0];
                    dy = this.brushStrokePoints[i + 1][1] - p[1];
                } else if (i > 0) {
                    dx = p[0] - this.brushStrokePoints[i - 1][0];
                    dy = p[1] - this.brushStrokePoints[i - 1][1];
                }
                const len = Math.hypot(dx, dy) || 1;
                const nx = -dy / len;
                const ny = dx / len;
                leftPts.push([p[0] + nx * r, p[1] + ny * r]);
                rightPts.unshift([p[0] - nx * r, p[1] - ny * r]);
            }
            boundary = leftPts.concat(rightPts);
        }

        boundary = this.simplifyPolygon(boundary);
        if (boundary.length < 3) return;

        if (this.activeTool === 'brush') {
            const polyObj = {
                id: 'poly_' + Date.now(),
                type: 'polygon',
                label: currentClass,
                polygon: boundary
            };
            this.core.annotations.objects.push(polyObj);
            this.core.selectedObjectId = polyObj.id;
        }
    }

    simplifyPolygon(points) {
        if (points.length <= 6) return points;
        const result = [];
        const step = Math.max(1, Math.floor(points.length / 32));
        for (let i = 0; i < points.length; i += step) {
            result.push(points[i]);
        }
        return result;
    }

    onContextMenu(pt, e) {
        // Right click closes current polygon
        if (this.currentPolygonPoints.length >= 3) {
            this.closePolygon();
        }
    }

    closePolygon() {
        if (this.currentPolygonPoints.length < 3) return;

        const currentClass = this.getSelectedClass() || 'object';
        const polyObj = {
            id: 'poly_' + Date.now(),
            type: 'polygon',
            label: currentClass,
            polygon: [...this.currentPolygonPoints]
        };

        this.core.annotations.objects.push(polyObj);
        this.core.selectedObjectId = polyObj.id;
        this.currentPolygonPoints = [];

        this.core.saveAnnotations();
        this.core.render();
    }

    isPointInPolygon(point, vs) {
        const x = point[0], y = point[1];
        let inside = false;
        for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
            const xi = vs[i][0], yi = vs[i][1];
            const xj = vs[j][0], yj = vs[j][1];
            const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    getColorForClass(label) {
        if (typeof window.stringToColor === 'function') {
            return window.stringToColor(label);
        }
        return '#00ff88';
    }

    render(ctx, annotations, selectedId) {
        const objects = annotations.objects || [];

        // Render Brush / Eraser Active Hover Cursor
        if ((this.activeTool === 'brush' || this.activeTool === 'eraser') && this.hoverPoint) {
            const radiusOnScreen = this.brushRadius;
            ctx.save();
            ctx.beginPath();
            ctx.arc(this.hoverPoint.x, this.hoverPoint.y, radiusOnScreen, 0, 2 * Math.PI);
            ctx.strokeStyle = (this.activeTool === 'brush') ? '#00f0ff' : '#ff3366';
            ctx.lineWidth = 1.5 / this.core.zoom;
            ctx.fillStyle = (this.activeTool === 'brush') ? 'rgba(0, 240, 255, 0.15)' : 'rgba(255, 51, 102, 0.15)';
            ctx.fill();
            ctx.stroke();
            ctx.restore();
        }

        // Render Active Painting Stroke
        if (this.isPainting && this.brushStrokePoints.length > 0) {
            ctx.save();
            ctx.beginPath();
            ctx.lineWidth = (this.brushRadius * 2);
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.strokeStyle = (this.activeTool === 'brush') ? 'rgba(0, 255, 136, 0.4)' : 'rgba(255, 0, 85, 0.4)';
            ctx.moveTo(this.brushStrokePoints[0][0], this.brushStrokePoints[0][1]);
            for (let i = 1; i < this.brushStrokePoints.length; i++) {
                ctx.lineTo(this.brushStrokePoints[i][0], this.brushStrokePoints[i][1]);
            }
            ctx.stroke();
            ctx.restore();
        }

        // 1. Render Existing Polygons
        for (let obj of objects) {
            if (obj.type !== 'polygon' || !obj.polygon || obj.polygon.length < 3) continue;
            const isSelected = obj.id === selectedId;
            const classColor = this.getColorForClass(obj.label);

            ctx.beginPath();
            ctx.moveTo(obj.polygon[0][0], obj.polygon[0][1]);
            for (let i = 1; i < obj.polygon.length; i++) {
                ctx.lineTo(obj.polygon[i][0], obj.polygon[i][1]);
            }
            ctx.closePath();

            // Fill & Stroke
            ctx.fillStyle = isSelected ? 'rgba(0, 240, 255, 0.35)' : 'rgba(0, 255, 136, 0.2)';
            ctx.fill();

            ctx.strokeStyle = isSelected ? '#00f0ff' : classColor;
            ctx.lineWidth = isSelected ? 3 / this.core.zoom : 2 / this.core.zoom;
            ctx.stroke();

            // Draw Vertices if Selected
            if (isSelected) {
                const nodeRadius = 5 / this.core.zoom;
                for (let i = 0; i < obj.polygon.length; i++) {
                    const pt = obj.polygon[i];
                    ctx.fillStyle = '#ffffff';
                    ctx.strokeStyle = '#00f0ff';
                    ctx.lineWidth = 1.5 / this.core.zoom;

                    ctx.beginPath();
                    ctx.arc(pt[0], pt[1], nodeRadius, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.stroke();
                }
            }
        }

        // 2. Render In-Progress Polygon Rubberband
        if (this.currentPolygonPoints.length > 0) {
            ctx.beginPath();
            ctx.moveTo(this.currentPolygonPoints[0][0], this.currentPolygonPoints[0][1]);
            for (let i = 1; i < this.currentPolygonPoints.length; i++) {
                ctx.lineTo(this.currentPolygonPoints[i][0], this.currentPolygonPoints[i][1]);
            }
            if (this.hoverPoint) {
                ctx.lineTo(this.hoverPoint.x, this.hoverPoint.y);
            }
            ctx.strokeStyle = '#ff0077';
            ctx.lineWidth = 2 / this.core.zoom;
            ctx.setLineDash([6 / this.core.zoom, 4 / this.core.zoom]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw vertices of in-progress polygon
            const nodeRadius = 5 / this.core.zoom;
            for (let i = 0; i < this.currentPolygonPoints.length; i++) {
                const pt = this.currentPolygonPoints[i];
                ctx.fillStyle = (i === 0) ? '#00ff88' : '#ff0077'; // First vertex green for snap target
                ctx.beginPath();
                ctx.arc(pt[0], pt[1], nodeRadius, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
    }
}
