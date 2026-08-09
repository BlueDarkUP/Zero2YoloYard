/**
 * DetectionAnnotator - Bounding Box Drawing & Rendering Plugin
 */
class DetectionAnnotator {
    constructor(core) {
        this.core = core;
        this.isDrawing = false;
        this.startX = 0;
        this.startY = 0;
        this.currentBox = null;
    }

    onMouseDown(pt, e) {
        this.isDrawing = true;
        this.startX = pt.x;
        this.startY = pt.y;
        this.currentBox = {
            id: 'bbox_' + Date.now(),
            type: 'bbox',
            label: this.core.selectedClass,
            bbox: [pt.x, pt.y, pt.x, pt.y]
        };
    }

    onMouseMove(pt, e) {
        if (!this.isDrawing || !this.currentBox) return;
        this.currentBox.bbox = [
            Math.min(this.startX, pt.x),
            Math.min(this.startY, pt.y),
            Math.max(this.startX, pt.x),
            Math.max(this.startY, pt.y)
        ];
        this.core.render();
    }

    onMouseUp(pt, e) {
        if (!this.isDrawing || !this.currentBox) return;
        this.isDrawing = false;
        const [x1, y1, x2, y2] = this.currentBox.bbox;
        if (Math.abs(x2 - x1) > 5 && Math.abs(y2 - y1) > 5) {
            this.core.annotations.objects.push(this.currentBox);
            this.core.saveAnnotations();
            if (this.core.onAnnotationsChanged) this.core.onAnnotationsChanged(this.core.annotations);
        }
        this.currentBox = null;
        this.core.render();
    }

    render(ctx, annotations, selectedId) {
        const objects = annotations.objects || [];
        for (let obj of objects) {
            if (obj.type !== 'bbox' || !obj.bbox) continue;
            const [x1, y1, x2, y2] = obj.bbox;
            const isSelected = obj.id === selectedId;

            ctx.strokeStyle = isSelected ? '#00f0ff' : '#00ff88';
            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            ctx.fillStyle = isSelected ? 'rgba(0, 240, 255, 0.2)' : 'rgba(0, 255, 136, 0.1)';
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);

            // Label tag
            ctx.fillStyle = '#000';
            ctx.fillRect(x1, y1 - 20, ctx.measureText(obj.label).width + 10, 20);
            ctx.fillStyle = '#00ff88';
            ctx.font = '12px "JetBrains Mono", monospace';
            ctx.fillText(obj.label, x1 + 5, y1 - 5);
        }

        // Draw active box
        if (this.currentBox) {
            const [x1, y1, x2, y2] = this.currentBox.bbox;
            ctx.strokeStyle = '#ff0077';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.setLineDash([]);
        }
    }
}
