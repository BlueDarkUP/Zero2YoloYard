# --- START OF FILE database.py ---

import json
import logging
import time
import uuid
import config
from bbox_writer import extract_labels

from sqlalchemy import create_engine, text, event
from sqlalchemy.pool import QueuePool

# ==============================================================================
# 数据库引擎与连接池配置
# ==============================================================================
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


# ==============================================================================
# 数据库迁移与初始化
# ==============================================================================

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

        add_column('datasets', 'eval_percent', 'REAL')
        add_column('datasets', 'test_percent', 'REAL')
        add_column('models', 'label_filename', 'TEXT')
        add_column('models', 'model_type', 'TEXT')
        add_column('videos', 'last_pre_annotation_info', 'TEXT')
        add_column('video_frames', 'tags', 'TEXT')
        add_column('video_frames', 'suggested_bboxes_text', 'TEXT')

        conn.execute(text('''
                          CREATE TABLE IF NOT EXISTS frame_labels
                          (
                              frame_id   INTEGER NOT NULL,
                              label_name TEXT    NOT NULL,
                              FOREIGN KEY (frame_id) REFERENCES video_frames (frame_id) ON DELETE CASCADE
                          )
                          '''))
        conn.execute(text('CREATE INDEX IF NOT EXISTS idx_frame_labels_name ON frame_labels (label_name)'))

        conn.execute(text('''
                          CREATE TABLE IF NOT EXISTS class_tags
                          (
                              tag_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                              tag_name       TEXT NOT NULL UNIQUE,
                              create_time_ms INTEGER
                          )
                          '''))


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
                              last_pre_annotation_info TEXT
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
                              test_percent      REAL
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


# ==============================================================================
# 数据操作 (CRUD) - 完美兼容原返回格式
# ==============================================================================

def create_video_entry(description, video_filename, file_size, create_time_ms):
    video_uuid = str(uuid.uuid4().hex)
    with engine.begin() as conn:
        conn.execute(
            text(
                'INSERT INTO videos (video_uuid, description, video_filename, file_size, create_time_ms, status) VALUES (:u, :d, :f, :s, :c, :st)'),
            {"u": video_uuid, "d": description, "f": video_filename, "s": file_size, "c": create_time_ms,
             "st": 'UPLOADING'}
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
        frames_to_insert = [{"u": video_uuid, "fn": i, "bt": "", "sb": "", "t": "", "inc": 1} for i in
                            range(frame_count)]

        # 批量插入提速
        conn.execute(
            text(
                'INSERT INTO video_frames (video_uuid, frame_number, bboxes_text, suggested_bboxes_text, tags, include_frame_in_dataset) VALUES (:u, :fn, :bt, :sb, :t, :inc)'),
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

        new_labeled_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM video_frames WHERE video_uuid = :u AND ((bboxes_text IS NOT NULL AND bboxes_text != '') OR (tags IS NOT NULL AND tags != '[]' AND tags != ''))"),
            {"u": video_uuid}
        ).scalar()

        conn.execute(text('UPDATE videos SET labeled_frame_count = :c WHERE video_uuid = :u'),
                     {"c": new_labeled_count, "u": video_uuid})


def save_frame_suggestions(video_uuid, frame_number, suggested_bboxes_text):
    with engine.begin() as conn:
        conn.execute(
            text('UPDATE video_frames SET suggested_bboxes_text = :sb WHERE video_uuid = :u AND frame_number = :fn'),
            {"sb": suggested_bboxes_text, "u": video_uuid, "fn": frame_number}
        )


def save_frame_tags(video_uuid, frame_number, tags_json_string):
    with engine.begin() as conn:
        conn.execute(
            text('UPDATE video_frames SET tags = :t WHERE video_uuid = :u AND frame_number = :fn'),
            {"t": tags_json_string, "u": video_uuid, "fn": frame_number}
        )
        new_labeled_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM video_frames WHERE video_uuid = :u AND ((bboxes_text IS NOT NULL AND bboxes_text != '') OR (tags IS NOT NULL AND tags != '[]' AND tags != ''))"),
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
            db_rows_to_insert.append({"u": video_uuid, "fn": new_frame_number, "bt": "", "sb": "", "t": "", "inc": 1})

        if db_rows_to_insert:
            conn.execute(
                text(
                    'INSERT INTO video_frames (video_uuid, frame_number, bboxes_text, suggested_bboxes_text, tags, include_frame_in_dataset) VALUES (:u, :fn, :bt, :sb, :t, :inc)'),
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


def create_dataset_entry(description, video_uuids, create_time_ms, eval_percent, test_percent):
    dataset_uuid = str(uuid.uuid4().hex)
    with engine.begin() as conn:
        conn.execute(
            text(
                'INSERT INTO datasets (dataset_uuid, description, video_uuids, create_time_ms, status, eval_percent, test_percent) VALUES (:du, :d, :vu, :c, :s, :ep, :tp)'),
            {"du": dataset_uuid, "d": description, "vu": json.dumps(video_uuids), "c": create_time_ms, "s": 'PENDING',
             "ep": eval_percent, "tp": test_percent}
        )
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
        return _to_dict(result)


def get_dataset_entity(dataset_uuid):
    with engine.connect() as conn:
        result = conn.execute(text('SELECT * FROM datasets WHERE dataset_uuid = :du'), {"du": dataset_uuid}).fetchone()
        return dict(result._mapping) if result else None


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