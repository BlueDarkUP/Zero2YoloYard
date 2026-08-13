import json
import logging
import time
import uuid
import config
import file_storage
from bbox_writer import extract_labels
from sqlalchemy import create_engine, text, event
from sqlalchemy.pool import QueuePool

# --- 数据库配置 ---
db_url = f"sqlite:///{config.DATABASE_FILE}"

# 创建带连接池的 SQLAlchemy 引擎
engine = create_engine(
    db_url,
    poolclass=QueuePool,
    pool_size=10,  # 保持10个常驻连接
    max_overflow=20,  # 峰值最多额外创建20个连接
    connect_args={
        'check_same_thread': False,  # 允许跨线程使用连接
        'timeout': 15  # 增加锁等待超时时间
    }
)


# 监听连接事件：开启 WAL 模式和性能调优
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    # 开启预写式日志，允许多个读操作和一个写操作并发执行
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    # NORMAL 在 WAL 模式下能提供极高写入性能，且安全性有保证
    cursor.execute("PRAGMA synchronous=NORMAL")
    # 增加内存缓存大小 (约为 64MB)
    cursor.execute("PRAGMA cache_size=-64000")
    # 存储临时表和索引在内存中
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


# 辅助函数：将 SQLAlchemy 的 Row 对象转为 dict，兼容老代码
def _to_dict(result_proxy):
    return [dict(row._mapping) for row in result_proxy]


# --- 数据库初始化与迁移 ---

def migrate_db():
    with engine.begin() as conn:
        # 获取现有的表和列
        def column_exists(table, column):
            result = conn.execute(text(f"PRAGMA table_info({table})"))
            return any(row[1] == column for row in result)

        def add_column(table, column, col_type):
            if not column_exists(table, column):
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                logging.info(f"Added column '{column}' to table '{table}'.")

        # --- 数据表自动补全列迁移 ---
        add_column('datasets', 'eval_percent', 'REAL')
        add_column('datasets', 'test_percent', 'REAL')
        # 补上下面这一行：自动为旧数据库补充 export_format 字段
        add_column('datasets', 'export_format', "TEXT DEFAULT 'yolo_v8_detect'")
        add_column('datasets', 'export_options', "TEXT DEFAULT '{}'")

        add_column('models', 'label_filename', 'TEXT')
        add_column('models', 'model_type', 'TEXT')
        add_column('videos', 'last_pre_annotation_info', 'TEXT')
        add_column('video_frames', 'tags', 'TEXT')
        add_column('video_frames', 'suggested_bboxes_text', 'TEXT')
        add_column('class_labels', 'sam3_prompt', 'TEXT')
        add_column('class_labels', 'keypoint_schema', 'TEXT')
        add_column('video_frames', 'annotations_json', 'TEXT')
        add_column('videos', 'annotation_type', 'TEXT')
        add_column('videos', 'keypoint_schema', 'TEXT')
        conn.execute(text("UPDATE videos SET annotation_type = 'detection' WHERE annotation_type IS NULL"))

        conn.execute(text('''
                          CREATE TABLE IF NOT EXISTS frame_labels
                          (
                              frame_id   INTEGER NOT NULL,
                              label_name TEXT    NOT NULL,
                              FOREIGN KEY (frame_id) REFERENCES video_frames (frame_id) ON DELETE CASCADE
                          )
                          '''))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_frame_labels_name ON frame_labels (label_name)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_frame_labels_frame_id ON frame_labels (frame_id)'))
        conn.execute(
            text('CREATE INDEX IF NOT EXISTS idx_video_frames_uuid_frame ON video_frames (video_uuid, frame_number)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_video_frames_uuid ON video_frames (video_uuid)'))

        conn.execute(text('''
                          CREATE TABLE IF NOT EXISTS class_tags
                          (
                              tag_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                              tag_name       TEXT NOT NULL UNIQUE,
                              create_time_ms INTEGER
                          )
                          '''))

        # 自动补全已有 annotations_json 帧的 frame_labels 和 class_labels 记录
        try:
            from annotation_model import AnnotationData
            rows = conn.execute(text(
                "SELECT frame_id, annotations_json FROM video_frames WHERE annotations_json IS NOT NULL AND TRIM(annotations_json) != ''"
            )).fetchall()
            for r in rows:
                fid = r[0]
                aj_str = r[1]
                ann_data = AnnotationData.from_json(aj_str)
                unique_labels = ann_data.get_unique_labels()
                if unique_labels:
                    conn.execute(text('DELETE FROM frame_labels WHERE frame_id = :fid'), {"fid": fid})
                    labels_to_insert = [{"fid": fid, "ln": label} for label in unique_labels]
                    conn.execute(text('INSERT INTO frame_labels (frame_id, label_name) VALUES (:fid, :ln)'), labels_to_insert)
                    for label in unique_labels:
                        conn.execute(
                            text('INSERT INTO class_labels (label_name, create_time_ms) VALUES (:ln, :c) ON CONFLICT(label_name) DO NOTHING'),
                            {"ln": label, "c": int(time.time() * 1000)}
                        )
            
            # 全局同步所有视频的 labeled_frame_count
            video_rows = conn.execute(text("SELECT video_uuid FROM videos")).fetchall()
            for (v_uuid,) in video_rows:
                count = conn.execute(
                    text("""
                        SELECT COUNT(DISTINCT vf.frame_id) FROM video_frames vf
                        WHERE vf.video_uuid = :u
                          AND (
                            vf.frame_id IN (SELECT DISTINCT frame_id FROM frame_labels)
                            OR (vf.bboxes_text IS NOT NULL AND TRIM(vf.bboxes_text) != '')
                            OR (vf.tags IS NOT NULL AND vf.tags != '[]' AND TRIM(vf.tags) != '')
                          )
                    """),
                    {"u": v_uuid}
                ).scalar()
                conn.execute(
                    text("UPDATE videos SET labeled_frame_count = :c WHERE video_uuid = :u"),
                    {"c": count, "u": v_uuid}
                )
        except Exception as e:
            logging.error(f"Error syncing existing annotations_json labels during migration: {e}")

        try:
            sug_rows = conn.execute(text(
                "SELECT video_uuid, frame_number, suggested_bboxes_text FROM video_frames WHERE suggested_bboxes_text IS NOT NULL AND TRIM(suggested_bboxes_text) != ''"
            )).fetchall()
        except Exception as e:
            sug_rows = []
            logging.error(f"Error reading historical suggested_bboxes_text: {e}")

    if sug_rows:
        try:
            for r in sug_rows:
                vu, fn, sb_text = r[0], r[1], r[2]
                convert_suggestions_to_formal_annotations(vu, fn, sb_text)
        except Exception as e:
            logging.error(f"Error migrating historical suggested_bboxes_text: {e}")

    force_resync_all_dataset_labels()


def force_resync_all_dataset_labels():
    """强行重新扫描所有帧的真实标注数据，修复视频标注计数与索引表"""
    logging.info("=== 开始对所有视频执行标注数据全量校准与修复... ===")
    try:
        import os
        import cv2
        from annotation_model import AnnotationData
        with engine.begin() as conn:
            # 1. 检查并修复视频状态与分辨率 (width/height)
            inter_statuses = ('EXTRACTING', 'PRE_ANNOTATING', 'APPLYING_CLASS', 'UPLOADING', 'CANCELLING')
            v_rows = conn.execute(text(
                "SELECT video_uuid, status, width, height, fps, frame_count, extracted_frame_count FROM videos"
            )).fetchall()

            for v_uuid, st, w, h, fps, fc, efc in v_rows:
                is_intermediate = (st in inter_statuses)
                needs_res = (w is None or h is None or w <= 0 or h <= 0)

                new_w, new_h = w, h
                new_fps, new_fc = fps, fc

                if is_intermediate or needs_res:
                    # 1.1 尝试从视频文件获取分辨率及视频基本元数据
                    vid_path = file_storage.get_video_path(v_uuid)
                    if os.path.exists(vid_path):
                        try:
                            cap = cv2.VideoCapture(vid_path)
                            if cap.isOpened():
                                cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                cap_fps = cap.get(cv2.CAP_PROP_FPS)
                                cap_fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                cap.release()
                                if cap_w > 0 and cap_h > 0:
                                    new_w, new_h = cap_w, cap_h
                                if cap_fps > 0 and (new_fps is None or new_fps <= 0):
                                    new_fps = cap_fps
                                if cap_fc > 0 and (new_fc is None or new_fc <= 0):
                                    new_fc = cap_fc
                        except Exception as e:
                            logging.warning(f"无法从视频文件 [{v_uuid[:8]}] 读取分辨率: {e}")

                    # 1.2 若从视频文件未获取到，尝试从抽取的帧图片读取分辨率
                    if new_w is None or new_h is None or new_w <= 0 or new_h <= 0:
                        frame_dir = file_storage.get_frame_dir(v_uuid)
                        if os.path.isdir(frame_dir):
                            for fname in os.listdir(frame_dir):
                                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    fpath = os.path.join(frame_dir, fname)
                                    try:
                                        img = cv2.imread(fpath)
                                        if img is not None:
                                            img_h, img_w = img.shape[:2]
                                            if img_w > 0 and img_h > 0:
                                                new_w, new_h = img_w, img_h
                                                break
                                    except Exception:
                                        pass

                    # 1.3 如果成功获取 width/height
                    if new_w is not None and new_h is not None and new_w > 0 and new_h > 0:
                        target_st = 'READY' if is_intermediate else st
                        update_params = {
                            "w": new_w,
                            "h": new_h,
                            "st": target_st,
                            "u": v_uuid
                        }
                        sql_parts = ["width = :w", "height = :h", "status = :st", "status_message = ''"]
                        if new_fps is not None and new_fps > 0:
                            sql_parts.append("fps = :fps")
                            update_params["fps"] = new_fps
                        if new_fc is not None and new_fc > 0:
                            sql_parts.append("frame_count = :fc")
                            update_params["fc"] = new_fc

                        conn.execute(text(f"UPDATE videos SET {', '.join(sql_parts)} WHERE video_uuid = :u"), update_params)
                        logging.info(f"视频 [{v_uuid[:8]}] 分辨率/状态校准成功: width={new_w}, height={new_h}, status={target_st}")
                    else:
                        # 1.4 无法获取 width/height，强制标记为 FAILED
                        conn.execute(text("""
                            UPDATE videos 
                            SET status = 'FAILED', status_message = 'Video upload or extraction incomplete (resolution width/height missing)' 
                            WHERE video_uuid = :u
                        """), {"u": v_uuid})
                        logging.warning(f"视频 [{v_uuid[:8]}] 缺少分辨率且无法恢复，状态更新为 FAILED")

                # 1.5 校准 extracted_frame_count 字段
                frame_dir = file_storage.get_frame_dir(v_uuid)
                if os.path.isdir(frame_dir):
                    actual_frames = len([f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    if actual_frames != efc:
                        conn.execute(text("UPDATE videos SET extracted_frame_count = :c WHERE video_uuid = :u"),
                                     {"c": actual_frames, "u": v_uuid})

            # 2. 遍历所有帧，从 JSON / 框文本 / 标签中解析真实标注
            rows = conn.execute(
                text("SELECT frame_id, video_uuid, bboxes_text, tags, annotations_json FROM video_frames")).fetchall()
            for r in rows:
                fid, vu, bt, t_str, aj_str = r[0], r[1], r[2], r[3], r[4]
                unique_labels = set()

                # 2.1 从传统框文本提取
                if bt and bt.strip():
                    unique_labels.update(extract_labels(bt))

                # 2.2 从 JSON 高级标注提取（包含分类 Tag、分割多边形、姿态等）
                if aj_str and aj_str.strip():
                    ann_data = AnnotationData.from_json(aj_str)
                    unique_labels.update(ann_data.get_unique_labels())

                # 2.3 从旧版 tags 字段提取
                if t_str and t_str.strip() and t_str != '[]':
                    try:
                        tags_list = json.loads(t_str)
                        unique_labels.update(tags_list)
                    except Exception:
                        pass

                # 2.4 同步补全 frame_labels 关联表
                if unique_labels:
                    conn.execute(text("DELETE FROM frame_labels WHERE frame_id = :fid"), {"fid": fid})
                    conn.execute(
                        text("INSERT INTO frame_labels (frame_id, label_name) VALUES (:fid, :ln)"),
                        [{"fid": fid, "ln": lbl} for lbl in unique_labels]
                    )
                    for lbl in unique_labels:
                        conn.execute(
                            text(
                                "INSERT INTO class_labels (label_name, create_time_ms) VALUES (:ln, :c) ON CONFLICT(label_name) DO NOTHING"),
                            {"ln": lbl, "c": int(time.time() * 1000)}
                        )

            # 3. 重新精确校准每一个视频的真实已标注帧数 labeled_frame_count
            video_rows = conn.execute(text("SELECT video_uuid FROM videos")).fetchall()
            for (v_uuid,) in video_rows:
                count = conn.execute(
                    text("""
                         SELECT COUNT(DISTINCT vf.frame_id)
                         FROM video_frames vf
                         WHERE vf.video_uuid = :u
                           AND (
                             vf.frame_id IN (SELECT DISTINCT frame_id FROM frame_labels)
                                 OR (vf.bboxes_text IS NOT NULL AND TRIM(vf.bboxes_text) != '')
                                 OR (vf.tags IS NOT NULL AND vf.tags != '[]' AND TRIM(vf.tags) != '')
                                 OR (vf.annotations_json IS NOT NULL AND TRIM(vf.annotations_json) != '' AND
                                     vf.annotations_json NOT LIKE '%"objects": [], "classifications": []%')
                             )
                         """),
                    {"u": v_uuid}
                ).scalar() or 0

                conn.execute(text("UPDATE videos SET labeled_frame_count = :c WHERE video_uuid = :u"),
                             {"c": count, "u": v_uuid})
                logging.info(f"视频 [{v_uuid[:8]}] 标注校准完成，实际已标注帧数: {count}")

        logging.info("=== 数据库全量数据修复与同步完成！ ===")
    except Exception as e:
        logging.error(f"校准数据库标注状态失败: {e}", exc_info=True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text('''
                          CREATE TABLE IF NOT EXISTS videos
                          (
                              video_uuid               TEXT PRIMARY KEY,
                              description              TEXT NOT NULL UNIQUE,
                              video_filename           TEXT,
                              file_size                INTEGER,
                              create_time_ms           INTEGER,
                              status                   TEXT,
                              status_message           TEXT,
                              width                    INTEGER,
                              height                   INTEGER,
                              fps                      REAL,
                              frame_count              INTEGER,
                              extracted_frame_count    INTEGER DEFAULT 0,
                              included_frame_count     INTEGER DEFAULT 0,
                              labeled_frame_count      INTEGER DEFAULT 0,
                              last_pre_annotation_info TEXT,
                              annotation_type          TEXT DEFAULT 'detection',
                              keypoint_schema          TEXT
                          )
                          '''))

        conn.execute(text('''
                          CREATE TABLE IF NOT EXISTS video_frames
                          (
                              frame_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                              video_uuid               TEXT,
                              frame_number             INTEGER,
                              bboxes_text              TEXT,
                              suggested_bboxes_text    TEXT,
                              tags                     TEXT,
                              include_frame_in_dataset INTEGER,
                              annotations_json         TEXT,
                              FOREIGN KEY (video_uuid) REFERENCES videos (video_uuid) ON DELETE CASCADE
                          )
                          '''))

        conn.execute(text('''
                          CREATE TABLE IF NOT EXISTS datasets
                          (
                              dataset_uuid      TEXT PRIMARY KEY,
                              description       TEXT NOT NULL UNIQUE,
                              video_uuids       TEXT,
                              create_time_ms    INTEGER,
                              status            TEXT,
                              status_message    TEXT,
                              zip_path          TEXT,
                              sorted_label_list TEXT,
                              eval_percent      REAL,
                              test_percent      REAL,
                              export_format     TEXT DEFAULT 'yolo_v8_detect'
                          )
                          '''))

        conn.execute(text('''
                          CREATE TABLE IF NOT EXISTS models
                          (
                              model_uuid     TEXT PRIMARY KEY,
                              description    TEXT NOT NULL UNIQUE,
                              create_time_ms INTEGER,
                              label_filename TEXT,
                              model_type     TEXT
                          )
                          '''))

        conn.execute(text('''
                          CREATE TABLE IF NOT EXISTS annotation_tasks
                          (
                              task_uuid      TEXT PRIMARY KEY,
                              video_uuid     TEXT    NOT NULL,
                              assigned_to    TEXT    NOT NULL,
                              description    TEXT,
                              start_frame    INTEGER NOT NULL,
                              end_frame      INTEGER NOT NULL,
                              status         TEXT,
                              create_time_ms INTEGER,
                              FOREIGN KEY (video_uuid) REFERENCES videos (video_uuid) ON DELETE CASCADE
                          )
                          '''))

        conn.execute(text('''
                          CREATE TABLE IF NOT EXISTS class_labels
                          (
                              label_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                              label_name     TEXT NOT NULL UNIQUE,
                              create_time_ms INTEGER
                          )
                          '''))

        conn.execute(text('''
                          CREATE TABLE IF NOT EXISTS class_tags
                          (
                              tag_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                              tag_name       TEXT NOT NULL UNIQUE,
                              create_time_ms INTEGER
                          )
                          '''))

        conn.execute(text('''
                          CREATE TABLE IF NOT EXISTS frame_labels
                          (
                              frame_id   INTEGER NOT NULL,
                              label_name TEXT    NOT NULL,
                              FOREIGN KEY (frame_id) REFERENCES video_frames (frame_id) ON DELETE CASCADE
                          )
                          '''))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_frame_labels_name ON frame_labels (label_name)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_frame_labels_frame_id ON frame_labels (frame_id)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_video_frames_uuid_frame ON video_frames (video_uuid, frame_number)'))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_video_frames_uuid ON video_frames (video_uuid)'))

    migrate_db()



# --- 数据操作 (CRUD) ---

def create_video_entry(description, video_filename, file_size, create_time_ms, annotation_type='detection'):
    video_uuid = str(uuid.uuid4().hex)
    with engine.begin() as conn:
        conn.execute(
            text(
                'INSERT INTO videos (video_uuid, description, video_filename, file_size, create_time_ms, status, annotation_type) VALUES (:u, :d, :f, :s, :c, :st, :at)'),
            {"u": video_uuid, "d": description, "f": video_filename, "s": file_size, "c": create_time_ms,
             "st": 'UPLOADING', "at": annotation_type}
        )
    return video_uuid


def get_ready_videos_with_labels():
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT * FROM videos WHERE status = 'READY' AND labeled_frame_count > 0 ORDER BY create_time_ms DESC"))
        return _to_dict(result)


def get_all_video_list():
    with engine.connect() as conn:
        result = conn.execute(text('SELECT * FROM videos ORDER BY create_time_ms DESC'))
        return _to_dict(result)


def get_video_entity(video_uuid):
    with engine.connect() as conn:
        result = conn.execute(text('SELECT * FROM videos WHERE video_uuid = :u'), {"u": video_uuid}).fetchone()
        return dict(result._mapping) if result else None


def update_video_status(video_uuid, status, message=""):
    with engine.begin() as conn:
        conn.execute(text('UPDATE videos SET status = :s, status_message = :m WHERE video_uuid = :u'),
                     {"s": status, "m": message, "u": video_uuid})


def update_pre_annotation_info(video_uuid, model_uuid, model_desc):
    info = {"model_uuid": model_uuid, "model_desc": model_desc, "time_ms": int(time.time() * 1000)}
    with engine.begin() as conn:
        conn.execute(text('UPDATE videos SET last_pre_annotation_info = :i WHERE video_uuid = :u'),
                     {"i": json.dumps(info), "u": video_uuid})


def update_video_after_extraction_start(video_uuid, width, height, fps, frame_count):
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE videos SET width=:w, height=:h, fps=:f, frame_count=:fc, included_frame_count=:fc, status=:st WHERE video_uuid=:u"),
            {"w": width, "h": height, "f": fps, "fc": frame_count, "st": 'EXTRACTING', "u": video_uuid}
        )
        frames_to_insert = [{"u": video_uuid, "fn": i, "bt": "", "sb": "", "t": "", "aj": "", "inc": 1} for i in
                            range(frame_count)]

        # 批量插入提速
        conn.execute(
            text(
                'INSERT INTO video_frames (video_uuid, frame_number, bboxes_text, suggested_bboxes_text, tags, annotations_json, include_frame_in_dataset) VALUES (:u, :fn, :bt, :sb, :t, :aj, :inc)'),
            frames_to_insert
        )


def update_extracted_frame_count(video_uuid, count):
    with engine.begin() as conn:
        conn.execute(text('UPDATE videos SET extracted_frame_count = :c WHERE video_uuid = :u'),
                     {"c": count, "u": video_uuid})


def delete_video(video_uuid):
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM videos WHERE video_uuid = :u'), {"u": video_uuid})


def get_video_frames(video_uuid):
    with engine.connect() as conn:
        result = conn.execute(text('SELECT * FROM video_frames WHERE video_uuid = :u ORDER BY frame_number ASC'),
                              {"u": video_uuid})
        return _to_dict(result)


def get_annotated_video_frames(video_uuid):
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT * FROM video_frames 
                WHERE video_uuid = :u 
                  AND (
                      frame_id IN (SELECT DISTINCT frame_id FROM frame_labels)
                      OR (bboxes_text IS NOT NULL AND TRIM(bboxes_text) != '')
                      OR (tags IS NOT NULL AND tags != '[]' AND TRIM(tags) != '')
                  ) 
                ORDER BY frame_number ASC
            """),
            {"u": video_uuid}
        )
        return _to_dict(result)

def save_frame_annotations(video_uuid, frame_number, annotations_json_str):
    with engine.begin() as conn:
        frame = conn.execute(
            text('SELECT frame_id FROM video_frames WHERE video_uuid = :u AND frame_number = :fn'),
            {"u": video_uuid, "fn": frame_number}
        ).fetchone()

        if not frame:
            logging.error(f"无法为 video {video_uuid}, frame {frame_number} 找到 frame_id。")
            return

        frame_id = frame._mapping['frame_id']
        conn.execute(
            text('UPDATE video_frames SET annotations_json = :aj WHERE frame_id = :fid'),
            {"aj": annotations_json_str, "fid": frame_id}
        )

        # 同步更新 frame_labels 和 class_labels 表
        conn.execute(text('DELETE FROM frame_labels WHERE frame_id = :fid'), {"fid": frame_id})
        from annotation_model import AnnotationData
        ann_data = AnnotationData.from_json(annotations_json_str)
        unique_labels = ann_data.get_unique_labels()
        if unique_labels:
            labels_to_insert = [{"fid": frame_id, "ln": label} for label in unique_labels]
            conn.execute(text('INSERT INTO frame_labels (frame_id, label_name) VALUES (:fid, :ln)'), labels_to_insert)

            for label in unique_labels:
                conn.execute(
                    text(
                        'INSERT INTO class_labels (label_name, create_time_ms) VALUES (:ln, :c) ON CONFLICT(label_name) DO NOTHING'),
                    {"ln": label, "c": int(time.time() * 1000)}
                )

        # 更新已标注帧计数
        new_labeled_count = conn.execute(
            text("""
                SELECT COUNT(DISTINCT vf.frame_id) FROM video_frames vf
                WHERE vf.video_uuid = :u 
                  AND (
                      vf.frame_id IN (SELECT DISTINCT frame_id FROM frame_labels)
                      OR (vf.bboxes_text IS NOT NULL AND TRIM(vf.bboxes_text) != '')
                      OR (vf.tags IS NOT NULL AND vf.tags != '[]' AND TRIM(vf.tags) != '')
                  )
            """),
            {"u": video_uuid}
        ).scalar()
        conn.execute(text('UPDATE videos SET labeled_frame_count = :c WHERE video_uuid = :u'),
                     {"c": new_labeled_count, "u": video_uuid})

def get_frame_annotations(video_uuid, frame_number):
    with engine.connect() as conn:
        result = conn.execute(
            text('SELECT annotations_json FROM video_frames WHERE video_uuid = :u AND frame_number = :fn'),
            {"u": video_uuid, "fn": frame_number}
        ).fetchone()
        if result and result._mapping['annotations_json']:
            import json
            try:
                return json.loads(result._mapping['annotations_json'])
            except:
                pass
        return None

def set_video_annotation_type(video_uuid, annotation_type):
    with engine.begin() as conn:
        conn.execute(text('UPDATE videos SET annotation_type = :t WHERE video_uuid = :u'),
                     {"t": annotation_type, "u": video_uuid})

def get_video_annotation_type(video_uuid):
    with engine.connect() as conn:
        result = conn.execute(text('SELECT annotation_type FROM videos WHERE video_uuid = :u'),
                              {"u": video_uuid}).fetchone()
        return result._mapping['annotation_type'] if result else 'detection'

def set_video_keypoint_schema(video_uuid, schema_json):
    with engine.begin() as conn:
        conn.execute(text('UPDATE videos SET keypoint_schema = :s WHERE video_uuid = :u'),
                     {"s": schema_json, "u": video_uuid})

def get_video_keypoint_schema(video_uuid):
    with engine.connect() as conn:
        result = conn.execute(text('SELECT keypoint_schema FROM videos WHERE video_uuid = :u'),
                              {"u": video_uuid}).fetchone()
        return result._mapping['keypoint_schema'] if result else None


def save_frame_bboxes(video_uuid, frame_number, bboxes_text):
    with engine.begin() as conn:
        frame = conn.execute(
            text('SELECT frame_id FROM video_frames WHERE video_uuid = :u AND frame_number = :fn'),
            {"u": video_uuid, "fn": frame_number}
        ).fetchone()

        if not frame:
            logging.error(f"无法为 video {video_uuid}, frame {frame_number} 找到 frame_id。")
            return

        frame_id = frame._mapping['frame_id']
        conn.execute(
            text('UPDATE video_frames SET bboxes_text = :bt, suggested_bboxes_text = :sb WHERE frame_id = :fid'),
            {"bt": bboxes_text, "sb": '', "fid": frame_id}
        )

        conn.execute(text('DELETE FROM frame_labels WHERE frame_id = :fid'), {"fid": frame_id})
        unique_labels = set(extract_labels(bboxes_text))
        if unique_labels:
            labels_to_insert = [{"fid": frame_id, "ln": label} for label in unique_labels]
            conn.execute(text('INSERT INTO frame_labels (frame_id, label_name) VALUES (:fid, :ln)'), labels_to_insert)

            for label in unique_labels:
                conn.execute(
                    text(
                        'INSERT INTO class_labels (label_name, create_time_ms) VALUES (:ln, :c) ON CONFLICT(label_name) DO NOTHING'),
                    {"ln": label, "c": int(time.time() * 1000)}
                )

        new_labeled_count = conn.execute(
            text("""
                SELECT COUNT(DISTINCT vf.frame_id) FROM video_frames vf
                WHERE vf.video_uuid = :u 
                  AND (
                      vf.frame_id IN (SELECT DISTINCT frame_id FROM frame_labels)
                      OR (vf.bboxes_text IS NOT NULL AND TRIM(vf.bboxes_text) != '')
                      OR (vf.tags IS NOT NULL AND vf.tags != '[]' AND TRIM(vf.tags) != '')
                  )
            """),
            {"u": video_uuid}
        ).scalar()

        conn.execute(text('UPDATE videos SET labeled_frame_count = :c WHERE video_uuid = :u'),
                     {"c": new_labeled_count, "u": video_uuid})


def convert_suggestions_to_formal_annotations(video_uuid, frame_number, suggested_bboxes_text):
    if not suggested_bboxes_text or not suggested_bboxes_text.strip():
        return

    annotation_type = get_video_annotation_type(video_uuid)
    lines = [line.strip() for line in suggested_bboxes_text.strip().split('\n') if line.strip()]
    if not lines:
        return

    if annotation_type == 'segmentation':
        from annotation_model import AnnotationData, AnnotationObject
        existing_ann_dict = get_frame_annotations(video_uuid, frame_number)
        if existing_ann_dict:
            ann_data = AnnotationData.from_dict(existing_ann_dict)
        else:
            ann_data = AnnotationData()

        for idx, line in enumerate(lines):
            parts = line.split(',')
            if len(parts) >= 5:
                xmin, ymin, xmax, ymax = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                label = parts[4]
                poly_str = parts[6] if len(parts) >= 7 else ""
                poly_pts = []
                if poly_str and '|' in poly_str:
                    try:
                        poly_pts = [[float(p.split(';')[0]), float(p.split(';')[1])] for p in poly_str.split('|') if ';' in p]
                    except:
                        poly_pts = []

                if not poly_pts or len(poly_pts) < 3:
                    poly_pts = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

                obj_id = f"poly_{int(time.time()*1000)}_{idx}"
                ann_data.objects.append(AnnotationObject(id=obj_id, type='polygon', label=label, points=poly_pts))

        save_frame_annotations(video_uuid, frame_number, ann_data.to_json())

    else:
        new_bbox_lines = []
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 5:
                xmin, ymin, xmax, ymax, label = parts[0], parts[1], parts[2], parts[3], parts[4]
                new_bbox_lines.append(f"{xmin},{ymin},{xmax},{ymax},{label}")

        existing = get_frame_bboxes(video_uuid, frame_number)
        if existing and existing.get('bboxes_text') and existing['bboxes_text'].strip():
            combined_text = existing['bboxes_text'].strip() + "\n" + "\n".join(new_bbox_lines)
        else:
            combined_text = "\n".join(new_bbox_lines)

        save_frame_bboxes(video_uuid, frame_number, combined_text)

    with engine.begin() as conn:
        conn.execute(
            text("UPDATE video_frames SET suggested_bboxes_text = '' WHERE video_uuid = :u AND frame_number = :fn"),
            {"u": video_uuid, "fn": frame_number}
        )

def save_frame_suggestions(video_uuid, frame_number, suggested_bboxes_text):
    convert_suggestions_to_formal_annotations(video_uuid, frame_number, suggested_bboxes_text)


def save_frame_tags(video_uuid, frame_number, tags_json_string):
    with engine.begin() as conn:
        conn.execute(
            text('UPDATE video_frames SET tags = :t WHERE video_uuid = :u AND frame_number = :fn'),
            {"t": tags_json_string, "u": video_uuid, "fn": frame_number}
        )
        new_labeled_count = conn.execute(
            text("""
                SELECT COUNT(DISTINCT vf.frame_id) FROM video_frames vf
                WHERE vf.video_uuid = :u 
                  AND (
                      vf.frame_id IN (SELECT DISTINCT frame_id FROM frame_labels)
                      OR (vf.bboxes_text IS NOT NULL AND TRIM(vf.bboxes_text) != '')
                      OR (vf.tags IS NOT NULL AND vf.tags != '[]' AND TRIM(vf.tags) != '')
                  )
            """),
            {"u": video_uuid}
        ).scalar()
        conn.execute(text('UPDATE videos SET labeled_frame_count = :c WHERE video_uuid = :u'),
                     {"c": new_labeled_count, "u": video_uuid})


def get_next_safe_frame_number(video_uuid):
    with engine.connect() as conn:
        max_val = conn.execute(text('SELECT MAX(frame_number) FROM video_frames WHERE video_uuid = :u'),
                               {"u": video_uuid}).scalar()
        return (max_val + 1) if max_val is not None else 0


def add_frames_to_video(video_uuid, frames_data_list):
    if not frames_data_list: return 0

    with engine.begin() as conn:
        start_frame_number = conn.execute(text('SELECT MAX(frame_number) FROM video_frames WHERE video_uuid = :u'),
                                          {"u": video_uuid}).scalar()
        start_frame_number = (start_frame_number + 1) if start_frame_number is not None else 0

        db_rows_to_insert = []
        for i, image_bytes in enumerate(frames_data_list):
            new_frame_number = start_frame_number + i
            file_storage.save_frame_image(video_uuid, new_frame_number, image_bytes)
            db_rows_to_insert.append({"u": video_uuid, "fn": new_frame_number, "bt": "", "sb": "", "t": "", "aj": "", "inc": 1})

        if db_rows_to_insert:
            conn.execute(
                text(
                    'INSERT INTO video_frames (video_uuid, frame_number, bboxes_text, suggested_bboxes_text, tags, annotations_json, include_frame_in_dataset) VALUES (:u, :fn, :bt, :sb, :t, :aj, :inc)'),
                db_rows_to_insert
            )

        final_count = conn.execute(text('SELECT COUNT(*) FROM video_frames WHERE video_uuid = :u'),
                                   {"u": video_uuid}).scalar()
        conn.execute(
            text(
                'UPDATE videos SET frame_count = :fc, extracted_frame_count = :fc, included_frame_count = :fc WHERE video_uuid = :u'),
            {"fc": final_count, "u": video_uuid}
        )
        return len(frames_data_list)


def add_frames_from_upload(video_uuid, frame_files):
    frames_data = [f.read() for f in frame_files]
    return add_frames_to_video(video_uuid, frames_data)


def create_annotation_task(video_uuid, assigned_to, description, start_frame, end_frame):
    task_uuid = str(uuid.uuid4().hex)
    create_time_ms = int(time.time() * 1000)

    with engine.begin() as conn:
        existing_tasks = conn.execute(text('SELECT start_frame, end_frame FROM annotation_tasks WHERE video_uuid = :u'),
                                      {"u": video_uuid}).fetchall()
        for task in existing_tasks:
            if start_frame <= task._mapping['end_frame'] and end_frame >= task._mapping['start_frame']:
                raise ValueError(
                    f"Frame range overlaps with an existing task ({task._mapping['start_frame']}-{task._mapping['end_frame']}).")

        conn.execute(
            text(
                'INSERT INTO annotation_tasks (task_uuid, video_uuid, assigned_to, description, start_frame, end_frame, status, create_time_ms) VALUES (:tu, :u, :a, :d, :sf, :ef, :st, :c)'),
            {"tu": task_uuid, "u": video_uuid, "a": assigned_to, "d": description, "sf": start_frame, "ef": end_frame,
             "st": 'PENDING', "c": create_time_ms}
        )
    return task_uuid


def get_tasks_for_video(video_uuid):
    with engine.connect() as conn:
        result = conn.execute(text('SELECT * FROM annotation_tasks WHERE video_uuid = :u ORDER BY create_time_ms DESC'),
                              {"u": video_uuid})
        return _to_dict(result)


def get_task_entity(task_uuid):
    with engine.connect() as conn:
        result = conn.execute(text('SELECT * FROM annotation_tasks WHERE task_uuid = :u'), {"u": task_uuid}).fetchone()
        return dict(result._mapping) if result else None


def delete_task(task_uuid):
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM annotation_tasks WHERE task_uuid = :u'), {"u": task_uuid})


def update_task_status(task_uuid, status):
    with engine.begin() as conn:
        conn.execute(text('UPDATE annotation_tasks SET status = :s WHERE task_uuid = :u'),
                     {"s": status, "u": task_uuid})


def add_class_label(label_name):
    with engine.begin() as conn:
        # SQLite 不支持 INSERT IGNORE，用 ON CONFLICT 避免报错
        conn.execute(
            text(
                'INSERT INTO class_labels (label_name, create_time_ms) VALUES (:ln, :c) ON CONFLICT(label_name) DO NOTHING'),
            {"ln": label_name, "c": int(time.time() * 1000)}
        )


def get_all_class_labels():
    with engine.connect() as conn:
        result = conn.execute(text('SELECT label_name FROM class_labels ORDER BY label_name ASC'))
        return [row[0] for row in result]


def get_all_class_labels_with_prompts():
    """
    给 SAM3 检索类功能（智能选择/LAM/批量应用/一致性检查）专用。
    返回 [{'label_name': ..., 'sam3_prompt': ...}]，sam3_prompt 可能为 None/空字符串
    （调用方需要自行 fallback 到 label_name，详见 ai_models.get_retrieval_text_for_class）。
    """
    with engine.connect() as conn:
        result = conn.execute(
            text('SELECT label_name, sam3_prompt FROM class_labels ORDER BY label_name ASC'))
        return [{'label_name': row[0], 'sam3_prompt': row[1]} for row in result]


def get_class_sam3_prompt(label_name):
    with engine.connect() as conn:
        result = conn.execute(
            text('SELECT sam3_prompt FROM class_labels WHERE label_name = :ln'), {"ln": label_name})
        row = result.fetchone()
        return row[0] if row else None


def set_class_sam3_prompt(label_name, sam3_prompt):
    """
    设置/更新某个类别用于 SAM3 检索的描述文本。传入空字符串或 None 会清空（即回退到用 label_name 本身）。
    """
    cleaned = (sam3_prompt or '').strip() or None
    now_ms = int(time.time() * 1000)
    with engine.begin() as conn:
        conn.execute(
            text('INSERT INTO class_labels (label_name, sam3_prompt, create_time_ms) VALUES (:ln, :sp, :c) '
                 'ON CONFLICT(label_name) DO UPDATE SET sam3_prompt = :sp'),
            {"sp": cleaned, "ln": label_name, "c": now_ms}
        )


def set_class_keypoint_schema(label_name, schema_json):
    """
    设置/更新某个类别的骨架点位与连线 Schema (JSON 字符串或 dict)。
    """
    if isinstance(schema_json, (dict, list)):
        schema_json = json.dumps(schema_json)
    cleaned = (schema_json or '').strip() or None
    now_ms = int(time.time() * 1000)
    with engine.begin() as conn:
        conn.execute(
            text('INSERT INTO class_labels (label_name, keypoint_schema, create_time_ms) VALUES (:ln, :ks, :c) '
                 'ON CONFLICT(label_name) DO UPDATE SET keypoint_schema = :ks'),
            {"ks": cleaned, "ln": label_name, "c": now_ms}
        )


def get_class_keypoint_schema(label_name):
    with engine.connect() as conn:
        res = conn.execute(
            text('SELECT keypoint_schema FROM class_labels WHERE label_name = :ln'),
            {"ln": label_name}
        ).fetchone()
        return res[0] if res else None


def get_all_class_keypoint_schemas():
    with engine.connect() as conn:
        res = conn.execute(text('SELECT label_name, keypoint_schema FROM class_labels')).fetchall()
        result_dict = {}
        for row in res:
            if row[1] and row[1].strip():
                try:
                    result_dict[row[0]] = json.loads(row[1])
                except Exception:
                    result_dict[row[0]] = row[1]
        return result_dict


def delete_class_label(label_name):
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM class_labels WHERE label_name = :ln'), {"ln": label_name})


def get_all_frames_with_class(class_name):
    query = """
            SELECT T1.*, T2.width, T2.height
            FROM video_frames AS T1
                     INNER JOIN videos AS T2 ON T1.video_uuid = T2.video_uuid
                     INNER JOIN frame_labels AS T3 ON T1.frame_id = T3.frame_id
            WHERE T3.label_name = :ln \
            """
    with engine.connect() as conn:
        result = conn.execute(text(query), {"ln": class_name})
        return _to_dict(result)


def add_class_tag(tag_name):
    with engine.begin() as conn:
        conn.execute(
            text('INSERT INTO class_tags (tag_name, create_time_ms) VALUES (:tn, :c) ON CONFLICT(tag_name) DO NOTHING'),
            {"tn": tag_name, "c": int(time.time() * 1000)}
        )


def get_all_class_tags():
    with engine.connect() as conn:
        result = conn.execute(text('SELECT tag_name FROM class_tags ORDER BY tag_name ASC'))
        return [row[0] for row in result]


def delete_class_tag(tag_name):
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM class_tags WHERE tag_name = :tn'), {"tn": tag_name})


def create_dataset_entry(description, video_uuids, create_time, eval_percent=20.0, test_percent=10.0, export_format='yolo_v8_detect', export_options=None):
    dataset_uuid = str(uuid.uuid4())
    video_uuids_json = json.dumps(video_uuids)
    if export_options is None:
        export_options = {}
    export_options_json = json.dumps(export_options) if isinstance(export_options, dict) else str(export_options)
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO datasets (dataset_uuid, description, video_uuids, create_time_ms, eval_percent, test_percent, export_format, export_options, status)
                VALUES (:dataset_uuid, :description, :video_uuids, :create_time, :eval_percent, :test_percent, :export_format, :export_options, 'PENDING')
            """),
            {
                "dataset_uuid": dataset_uuid,
                "description": description,
                "video_uuids": video_uuids_json,
                "create_time": create_time,
                "eval_percent": eval_percent,
                "test_percent": test_percent,
                "export_format": export_format,
                "export_options": export_options_json,
            }
        )
        conn.commit()
    return dataset_uuid


def update_dataset_status(dataset_uuid, status, message="", zip_path="", sorted_label_list=None):
    with engine.begin() as conn:
        if sorted_label_list is not None:
            conn.execute(
                text(
                    'UPDATE datasets SET status=:s, status_message=:m, zip_path=:z, sorted_label_list=:sl WHERE dataset_uuid=:du'),
                {"s": status, "m": message, "z": zip_path, "sl": json.dumps(sorted_label_list), "du": dataset_uuid}
            )
        else:
            conn.execute(
                text('UPDATE datasets SET status=:s, status_message=:m, zip_path=:z WHERE dataset_uuid=:du'),
                {"s": status, "m": message, "z": zip_path, "du": dataset_uuid}
            )


def get_dataset_list():
    with engine.connect() as conn:
        result = conn.execute(text('SELECT * FROM datasets ORDER BY create_time_ms DESC'))
        datasets = _to_dict(result)
        for d in datasets:
            if d.get('export_options') and isinstance(d['export_options'], str):
                try:
                    d['export_options'] = json.loads(d['export_options'])
                except Exception:
                    d['export_options'] = {}
            elif not d.get('export_options'):
                d['export_options'] = {}
        return datasets


get_all_datasets = get_dataset_list


def get_dataset_entity(dataset_uuid):
    with engine.connect() as conn:
        result = conn.execute(text('SELECT * FROM datasets WHERE dataset_uuid = :du'), {"du": dataset_uuid}).fetchone()
        if not result:
            return None
        d = dict(result._mapping)
        if d.get('export_options') and isinstance(d['export_options'], str):
            try:
                d['export_options'] = json.loads(d['export_options'])
            except Exception:
                d['export_options'] = {}
        elif not d.get('export_options'):
            d['export_options'] = {}
        return d


def delete_dataset(dataset_uuid):
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM datasets WHERE dataset_uuid = :du'), {"du": dataset_uuid})


def import_model_metadata(description, label_filename, model_type, create_time_ms):
    model_uuid = str(uuid.uuid4().hex)
    with engine.begin() as conn:
        conn.execute(
            text(
                'INSERT INTO models (model_uuid, description, label_filename, model_type, create_time_ms) VALUES (:mu, :d, :lf, :mt, :c)'),
            {"mu": model_uuid, "d": description, "lf": label_filename, "mt": model_type, "c": create_time_ms}
        )
    return model_uuid


def get_model_list():
    with engine.connect() as conn:
        result = conn.execute(text('SELECT * FROM models ORDER BY create_time_ms DESC'))
        return _to_dict(result)


def get_model_entity(model_uuid):
    with engine.connect() as conn:
        result = conn.execute(text('SELECT * FROM models WHERE model_uuid = :mu'), {"mu": model_uuid}).fetchone()
        return dict(result._mapping) if result else None


def delete_model(model_uuid):
    with engine.begin() as conn:
        conn.execute(text('DELETE FROM models WHERE model_uuid = :mu'), {"mu": model_uuid})


def get_frame_numbers_for_video(video_uuid):
    with engine.connect() as conn:
        result = conn.execute(
            text('SELECT frame_number FROM video_frames WHERE video_uuid = :u ORDER BY frame_number ASC'),
            {"u": video_uuid})
        return [row[0] for row in result]

def get_frame_bboxes(video_uuid, frame_number):
    """获取指定帧的边界框数据"""
    with engine.connect() as conn:
        result = conn.execute(
            text('SELECT bboxes_text FROM video_frames WHERE video_uuid = :u AND frame_number = :fn'),
            {"u": video_uuid, "fn": frame_number}
        ).fetchone()
        return dict(result._mapping) if result else None

def get_class_usage_count(label_name):
    """统计某个类别在全局多少个帧中被使用了"""
    try:
        with engine.connect() as conn:
            # 查询 frame_labels 表中该类别的出现次数
            count = conn.execute(
                text('SELECT COUNT(DISTINCT frame_id) FROM frame_labels WHERE label_name = :ln'),
                {"ln": label_name}
            ).scalar()
            return count or 0
    except Exception as e:
        logging.error(f"Error counting class usage for {label_name}: {e}")
        return 0