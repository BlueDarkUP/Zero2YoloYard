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

        // Key listeners for closing, canceling, or deleting polygon
        window.addEventListener('keydown', (e) => {
            if ($(e.target).is('input, textarea')) return;

            if (e.key === 'Enter') {
                this.closePolygon();
            } else if (e.key === 'Escape') {
                if (this.currentPolygonPoints.length > 0) {
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
        if (this.draggedVertexIndex !== null) {
            this.draggedVertexIndex = null;
            this.core.saveAnnotations();
            this.core.render();
        }
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
