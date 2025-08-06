# interactive_bbox_item.py
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsTextItem, QGraphicsItem, QStyleOptionGraphicsItem, QWidget, \
    QMenu
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QCursor
from PySide6.QtCore import Qt, QRectF, QPointF


class InteractiveBBoxItem(QGraphicsRectItem):
    handle_size = 10.0

    def __init__(self, rect: QRectF, annotation_data, class_name: str, class_color: QColor,
                 parent: QGraphicsItem = None):
        super().__init__(QRectF(0, 0, rect.width(), rect.height()), parent)
        self.setPos(rect.topLeft())

        self.annotation_data = annotation_data
        self.class_name = class_name
        self.class_color = class_color

        self.handles = {}
        self.handle_selected = None
        self.mouse_press_pos_scene = None
        self.mouse_press_rect = None

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)

        self.text_item = QGraphicsTextItem(self._get_display_text(), self)
        self.text_item.setDefaultTextColor(Qt.white)
        self.update_text_pos()

        self.update_handles_pos()

    def _get_display_text(self):
        text = f"{self.annotation_data.class_id}: {self.class_name}"
        if self.annotation_data.track_id is not None:
            track_info = f" (ID:{self.annotation_data.track_id}"
            if self.annotation_data.track_name:
                track_info += f", Name:{self.annotation_data.track_name}"
            track_info += ")"
            text += track_info
        return text

    def update_text_pos(self):
        self.text_item.setPos(0, -self.text_item.boundingRect().height())

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            pass
        return super().itemChange(change, value)

    def handle_at(self, pos_item: QPointF):
        for handle, rect in self.handles.items():
            if rect.contains(pos_item):
                return handle
        return None

    def hoverMoveEvent(self, event):
        handle = self.handle_at(event.pos())
        if handle:
            if handle in ['top_left', 'bottom_right']:
                self.setCursor(Qt.SizeFDiagCursor)
            elif handle in ['top_right', 'bottom_left']:
                self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.SizeAllCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if not self.isSelected():
            scene = self.scene()
            if scene:
                for item in scene.items():
                    if item != self and isinstance(item, InteractiveBBoxItem):
                        item.setSelected(False)
            self.setSelected(True)

        self.mouse_press_pos_scene = event.scenePos()
        self.handle_selected = self.handle_at(event.pos())
        if self.handle_selected:
            self.mouse_press_rect = self.rect()
            self.setSelected(True)
        else:
            super().mousePressEvent(event)
            event.accept()

    def mouseMoveEvent(self, event):
        if self.handle_selected:
            self.prepareGeometryChange()

            current_pos_scene = event.scenePos()
            delta_scene = current_pos_scene - self.mouse_press_pos_scene

            new_rect = QRectF(self.mouse_press_rect)

            if 'left' in self.handle_selected:
                new_pos_x = self.pos().x() + delta_scene.x()
                new_width = self.rect().width() - delta_scene.x()
                if new_width > self.handle_size:
                    self.setX(new_pos_x)
                    new_rect.setWidth(new_width)

            if 'right' in self.handle_selected:
                new_width = self.rect().width() + delta_scene.x()
                if new_width > self.handle_size:
                    new_rect.setWidth(new_width)

            if 'top' in self.handle_selected:
                new_pos_y = self.pos().y() + delta_scene.y()
                new_height = self.rect().height() - delta_scene.y()
                if new_height > self.handle_size:
                    self.setY(new_pos_y)
                    new_rect.setHeight(new_height)

            if 'bottom' in self.handle_selected:
                new_height = self.rect().height() + delta_scene.y()
                if new_height > self.handle_size:
                    new_rect.setHeight(new_height)

            self.setRect(new_rect)
            self.update_handles_pos()
            self.update_text_pos()
            self.mouse_press_pos_scene = current_pos_scene

        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.handle_selected = None
        self.mouse_press_pos_scene = None
        self.mouse_press_rect = None
        self.update()

    def contextMenuEvent(self, event):
        menu = QMenu()

        if hasattr(self.scene().views()[0], 'parent_window') and \
                hasattr(self.scene().views()[0].parent_window, 'show_bbox_context_menu'):
            self.scene().views()[0].parent_window.show_bbox_context_menu(event.screenPos())

        event.accept()

    def update_handles_pos(self):
        s = self.handle_size
        r = self.rect()
        self.handles['top_left'] = QRectF(r.left(), r.top(), s, s).translated(-s / 2, -s / 2)
        self.handles['top_right'] = QRectF(r.right(), r.top(), s, s).translated(-s / 2, -s / 2)
        self.handles['bottom_left'] = QRectF(r.left(), r.bottom(), s, s).translated(-s / 2, -s / 2)
        self.handles['bottom_right'] = QRectF(r.right(), r.bottom(), s, s).translated(-s / 2, -s / 2)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget):
        pen = QPen(self.class_color, 2)
        if self.isSelected():
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self.rect())

        text_bg_rect = self.text_item.boundingRect()
        text_bg_rect.moveTopLeft(self.text_item.pos())
        painter.setBrush(QBrush(self.class_color.darker(150)))
        painter.setPen(Qt.NoPen)
        painter.drawRect(text_bg_rect)

        if self.isSelected():
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QBrush(QColor(255, 255, 255, 150)))
            painter.setPen(QPen(self.class_color, 1.0, Qt.SolidLine))
            for rect in self.handles.values():
                painter.drawEllipse(rect)