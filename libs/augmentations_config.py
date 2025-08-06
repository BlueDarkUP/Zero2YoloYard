# augmentations_config.py
import albumentations as A

AUGMENTATIONS = {
    "几何变换": {
        "水平翻转 (HorizontalFlip)": {
            "class": A.HorizontalFlip,
            "type": "geometric",
            "params": {"p": {"type": "float", "default": 0.5, "range": (0.0, 1.0), "label": "应用概率"}}
        },
        "垂直翻转 (VerticalFlip)": {
            "class": A.VerticalFlip,
            "type": "geometric",
            "params": {"p": {"type": "float", "default": 0.0, "range": (0.0, 1.0), "label": "应用概率"}}
        },
        "旋转 (Rotate)": {
            "class": A.Rotate,
            "type": "geometric",
            "params": {
                "limit": {"type": "range", "default": (-15, 15), "range": (-180, 180), "label": "角度范围"},
                "p": {"type": "float", "default": 0.5, "range": (0.0, 1.0), "label": "应用概率"}
            }
        },
        "缩放与位移 (ShiftScaleRotate)": {
            "class": A.ShiftScaleRotate,
            "type": "geometric",
            "params": {
                "shift_limit": {"type": "float", "default": 0.06, "range": (0.0, 0.5), "label": "位移限制"},
                "scale_limit": {"type": "float", "default": 0.1, "range": (0.0, 0.5), "label": "缩放限制"},
                "rotate_limit": {"type": "int", "default": 15, "range": (0, 180), "label": "旋转限制(度)"},
                "p": {"type": "float", "default": 0.5, "range": (0.0, 1.0), "label": "应用概率"}
            }
        },
        "随机尺寸裁切 (RandomResizedCrop)": {
            "class": A.RandomResizedCrop,
            "type": "geometric",
            "params": {
                # 关键修复点: 移除UI不支持的 'range_float' 类型，让 'scale' 使用默认值。
                # 'height' 和 'width' 作为静态值，现在可以被正确读取了。
                "height": {"type": "int_static", "default": 640, "label": "目标高度"},
                "width": {"type": "int_static", "default": 640, "label": "目标宽度"},
                "p": {"type": "float", "default": 0.2, "range": (0.0, 1.0), "label": "应用概率"}
            }
        },
    },
    "色彩与亮度": {
        "灰度 (Grayscale)": {
            "class": A.ToGray,
            "type": "pixel",
            "params": {"p": {"type": "float", "default": 0.1, "range": (0.0, 1.0), "label": "应用概率"}}
        },
        "色彩抖动 (ColorJitter)": {
            "class": A.ColorJitter,
            "type": "pixel",
            "params": {
                "brightness": {"type": "float", "default": 0.2, "range": (0.0, 1.0), "label": "亮度"},
                "contrast": {"type": "float", "default": 0.2, "range": (0.0, 1.0), "label": "对比度"},
                "saturation": {"type": "float", "default": 0.2, "range": (0.0, 1.0), "label": "饱和度"},
                "hue": {"type": "float", "default": 0.1, "range": (0.0, 0.5), "label": "色调"},
                "p": {"type": "float", "default": 0.8, "range": (0.0, 1.0), "label": "应用概率"}
            }
        },
        "随机亮度和对比度": {
            "class": A.RandomBrightnessContrast,
            "type": "pixel",
            "params": {
                "brightness_limit": {"type": "float", "default": 0.2, "range": (0.0, 1.0), "label": "亮度限制"},
                "contrast_limit": {"type": "float", "default": 0.2, "range": (0.0, 1.0), "label": "对比度限制"},
                "p": {"type": "float", "default": 0.5, "range": (0.0, 1.0), "label": "应用概率"}
            }
        },
        "曝光 (RandomGamma)": {
            "class": A.RandomGamma,
            "type": "pixel",
            "params": {"p": {"type": "float", "default": 0.3, "range": (0.0, 1.0), "label": "应用概率"}}
        },
    },
    "模糊与噪声": {
        "模糊 (Blur)": {
            "class": A.Blur,
            "type": "pixel",
            "params": {
                "blur_limit": {"type": "int", "default": 7, "range": (3, 21), "label": "模糊限制"},
                "p": {"type": "float", "default": 0.3, "range": (0.0, 1.0), "label": "应用概率"}
            }
        },
        "高斯噪声 (GaussNoise)": {
            "class": A.GaussNoise,
            "type": "pixel",
            "params": {"p": {"type": "float", "default": 0.2, "range": (0.0, 1.0), "label": "应用概率"}}
        },
        "运动模糊 (MotionBlur)": {
            "class": A.MotionBlur,
            "type": "pixel",
            "params": {"p": {"type": "float", "default": 0.2, "range": (0.0, 1.0), "label": "应用概率"}},
        }
    },
    "遮挡": {
        "随机擦除 (Cutout/CoarseDropout)": {
            "class": A.CoarseDropout,
            "type": "pixel",
             "params": {
                "max_holes": {"type": "int", "default": 8, "range": (1, 16), "label": "最大孔洞数"},
                "max_height": {"type": "int", "default": 32, "range": (8, 128), "label": "最大高度"},
                "max_width": {"type": "int", "default": 32, "range": (8, 128), "label": "最大宽度"},
                "p": {"type": "float", "default": 0.5, "range": (0.0, 1.0), "label": "应用概率"}
            }
        }
    }
}