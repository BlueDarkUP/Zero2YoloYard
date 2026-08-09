/**
 * ClassificationAnnotator - Image & Frame Classification Annotator Plugin
 * Supports 1-Key Quick Tagging (Keys 1-9), Auto-Advance, & Canvas Banner Overlays.
 */
class ClassificationAnnotator {
    constructor(core) {
        this.core = core;
        this.autoAdvance = true;

        this.initDOM();
        this.bindShortcuts();
    }

    initDOM() {
        const self = this;

        // Render Search Bar and Tag Chips
        $(document).off('input', '#classification-search').on('input', '#classification-search', function() {
            const query = $(this).val().toLowerCase();
            $('#classification-tag-container .class-chip').each(function() {
                const label = $(this).data('class-name').toLowerCase();
                $(this).toggle(label.includes(query));
            });
        });

        // Click Tag Chip -> Assign Classification
        $(document).off('click', '#classification-tag-container .class-chip').on('click', '#classification-tag-container .class-chip', function() {
            const className = $(this).data('class-name');
            self.setClassification(className);
        });

        // Render Class Chips initially
        this.renderSidebarTags();
    }

    bindShortcuts() {
        const self = this;
        window.addEventListener('keydown', (e) => {
            if ($(e.target).is('input, textarea')) return;

            // Number keys 1-9 for quick classification
            const num = parseInt(e.key);
            if (!isNaN(num) && num >= 1 && num <= 9) {
                const availableClasses = self.getAvailableClasses();
                if (num <= availableClasses.length) {
                    const targetClass = availableClasses[num - 1];
                    e.preventDefault();
                    self.setClassification(targetClass);
                }
            } else if (e.key === 'c' || e.key === 'C') {
                // 'C' to clear classification
                self.clearClassification();
            }
        });
    }

    getAvailableClasses() {
        if (typeof window.availableClasses !== 'undefined' && Array.isArray(window.availableClasses) && window.availableClasses.length > 0) {
            return window.availableClasses;
        }
        const classes = [];
        $('#class-list li').each(function() {
            const cls = $(this).data('class-name');
            if (cls) classes.push(cls);
        });
        return classes;
    }

    setClassification(className) {
        if (!this.core.annotations.classifications) {
            this.core.annotations.classifications = [];
        }

        const idx = this.core.annotations.classifications.indexOf(className);
        if (idx !== -1) {
            this.core.annotations.classifications.splice(idx, 1);
            if (typeof window.showToast === 'function') {
                window.showToast(`🗑️ Removed [${className}]`, 1200);
            }
        } else {
            this.core.annotations.classifications.push(className);
            if (typeof window.showToast === 'function') {
                window.showToast(`🏷️ Classified as [${className}]`, 1200);
            }
        }

        this.core.saveAnnotations();
        this.renderSidebarTags();
        this.core.render();

    }

    clearClassification() {
        this.core.annotations.classifications = [];
        this.core.saveAnnotations();
        this.renderSidebarTags();
        this.core.render();
        if (typeof window.showToast === 'function') {
            window.showToast(`🗑️ Cleared classification`, 1200);
        }
    }

    renderSidebarTags() {
        const container = $('#classification-tag-container');
        if (!container.length) return;

        const classes = this.getAvailableClasses();
        const currentClasses = this.core.annotations.classifications || [];

        container.empty();

        if (classes.length === 0) {
            container.html('<div class="text-muted small">No classes configured. Add classes in the sidebar.</div>');
            return;
        }

        classes.forEach((cls, idx) => {
            const isAssigned = currentClasses.includes(cls);
            const shortcutNum = (idx < 9) ? `[${idx + 1}] ` : '';
            const chip = $(`
                <div class="class-chip px-3 py-2 border rounded text-truncate font-weight-bold ${isAssigned ? 'bg-warning text-dark border-warning' : 'bg-dark text-light border-secondary'}" 
                     data-class-name="${cls}" 
                     style="cursor: pointer; transition: all 0.2s ease; font-size: 0.85rem; user-select: none;">
                    <span class="opacity-75 mr-1 font-monospace" style="font-size: 0.75rem;">${shortcutNum}</span>
                    <span>${cls}</span>
                    ${isAssigned ? '<i class="bi bi-check-circle-fill ml-2 text-dark"></i>' : ''}
                </div>
            `);
            container.append(chip);
        });
    }

    render(ctx, annotations) {
        // Sync sidebar tag active states
        this.renderSidebarTags();

        const classifications = annotations.classifications || [];
        const isLabeled = classifications.length > 0;

        ctx.save();

        // Render Top-Left Canvas Banner Badge
        const badgeX = 20 / this.core.zoom;
        const badgeY = 20 / this.core.zoom;
        const fontSize = Math.max(14, 18 / this.core.zoom);

        ctx.font = `bold ${fontSize}px "Inter", sans-serif`;

        const labelText = isLabeled 
            ? `🏷️ CLASS: [ ${classifications.join(', ')} ]`
            : `⚠️ UNLABELED (Press 1-9 to tag)`;

        const textWidth = ctx.measureText(labelText).width;
        const paddingH = 14 / this.core.zoom;
        const paddingV = 8 / this.core.zoom;
        const boxW = textWidth + paddingH * 2;
        const boxH = fontSize + paddingV * 2;

        // Draw Badge Background
        ctx.fillStyle = isLabeled ? 'rgba(0, 255, 136, 0.9)' : 'rgba(255, 180, 0, 0.9)';
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(badgeX, badgeY, boxW, boxH, 6 / this.core.zoom) : ctx.rect(badgeX, badgeY, boxW, boxH);
        ctx.fill();

        // Draw Badge Text
        ctx.fillStyle = '#000000';
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, badgeX + paddingH, badgeY + boxH / 2);

        ctx.restore();
    }
}
