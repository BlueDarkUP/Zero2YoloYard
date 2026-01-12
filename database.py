import json
import logging
import sqlite3
import time
import uuid
import config
import file_storage
from bbox_writer import extract_labels


def get_db_connection():
    conn = sqlite3.connect(config.DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_not_exists(cursor, table_name, column_name, column_type):
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row['name'] for row in cursor.fetchall()]
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            print(f"Added column '{column_name}' to table '{table_name}'.")
    except Exception as e:
        print(f"Migration warning for {table_name}.{column_name}: {e}")


def migrate_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. 补全缺失的列
    _add_column_if_not_exists(cursor, 'datasets', 'eval_percent', 'REAL')
    _add_column_if_not_exists(cursor, 'datasets', 'test_percent', 'REAL')
    _add_column_if_not_exists(cursor, 'models', 'label_filename', 'TEXT')
    _add_column_if_not_exists(cursor, 'models', 'model_type', 'TEXT')
    _add_column_if_not_exists(cursor, 'videos', 'last_pre_annotation_info', 'TEXT')
    _add_column_if_not_exists(cursor, 'video_frames', 'tags', 'TEXT')
    _add_column_if_not_exists(cursor, 'video_frames', 'suggested_bboxes_text', 'TEXT')

    # 2. 补全缺失的表 (针对旧版本升级用户)
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS frame_labels
                   (
                       frame_id   INTEGER NOT NULL,
                       label_name TEXT    NOT NULL,
                       FOREIGN KEY (frame_id) REFERENCES video_frames (frame_id) ON DELETE CASCADE
                   )
                   ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_frame_labels_name ON frame_labels (label_name)')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS class_tags
                   (
                       tag_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                       tag_name       TEXT NOT NULL UNIQUE,
                       create_time_ms INTEGER
                   )
                   ''')

    conn.commit()
    conn.close()


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
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
                   ''')

    cursor.execute('''
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
                   ''')

    cursor.execute('''
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
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS models
                   (
                       model_uuid     TEXT PRIMARY KEY,
                       description    TEXT NOT NULL UNIQUE,
                       create_time_ms INTEGER,
                       label_filename TEXT,
                       model_type     TEXT
                   )
                   ''')

    cursor.execute('''
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
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS class_labels
                   (
                       label_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                       label_name     TEXT NOT NULL UNIQUE,
                       create_time_ms INTEGER
                   )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS class_tags
                   (
                       tag_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                       tag_name       TEXT NOT NULL UNIQUE,
                       create_time_ms INTEGER
                   )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS frame_labels
                   (
                       frame_id   INTEGER NOT NULL,
                       label_name TEXT    NOT NULL,
                       FOREIGN KEY (frame_id) REFERENCES video_frames (frame_id) ON DELETE CASCADE
                   )
                   ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_frame_labels_name ON frame_labels (label_name)')

    conn.commit()
    conn.close()


def create_video_entry(description, video_filename, file_size, create_time_ms):
    conn = get_db_connection()
    video_uuid = str(uuid.uuid4().hex)
    conn.execute(
        'INSERT INTO videos (video_uuid, description, video_filename, file_size, create_time_ms, status) VALUES (?, ?, ?, ?, ?, ?)',
        (video_uuid, description, video_filename, file_size, create_time_ms, 'UPLOADING')
    )
    conn.commit()
    conn.close()
    return video_uuid


def get_ready_videos_with_labels():
    conn = get_db_connection()
    # 修复：使用单引号 'READY' 避免被识别为列名
    videos = conn.execute(
        "SELECT * FROM videos WHERE status = 'READY' AND labeled_frame_count > 0 ORDER BY create_time_ms DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in videos]


def get_all_video_list():
    conn = get_db_connection()
    videos = conn.execute('SELECT * FROM videos ORDER BY create_time_ms DESC').fetchall()
    conn.close()
    return [dict(row) for row in videos]


def get_video_entity(video_uuid):
    conn = get_db_connection()
    video = conn.execute('SELECT * FROM videos WHERE video_uuid = ?', (video_uuid,)).fetchone()
    conn.close()
    return dict(video) if video else None


def update_video_status(video_uuid, status, message=""):
    conn = get_db_connection()
    conn.execute('UPDATE videos SET status = ?, status_message = ? WHERE video_uuid = ?', (status, message, video_uuid))
    conn.commit()
    conn.close()


def update_pre_annotation_info(video_uuid, model_uuid, model_desc):
    conn = get_db_connection()
    info = {
        "model_uuid": model_uuid,
        "model_desc": model_desc,
        "time_ms": int(time.time() * 1000)
    }
    conn.execute('UPDATE videos SET last_pre_annotation_info = ? WHERE video_uuid = ?', (json.dumps(info), video_uuid))
    conn.commit()
    conn.close()


def update_video_after_extraction_start(video_uuid, width, height, fps, frame_count):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE videos SET width=?, height=?, fps=?, frame_count=?, included_frame_count=?, status=? WHERE video_uuid=?",
        (width, height, fps, frame_count, frame_count, 'EXTRACTING', video_uuid)
    )
    frames_to_insert = [(video_uuid, i, '', '', '', 1) for i in range(frame_count)]
    cursor.executemany(
        'INSERT INTO video_frames (video_uuid, frame_number, bboxes_text, suggested_bboxes_text, tags, include_frame_in_dataset) VALUES (?, ?, ?, ?, ?, ?)',
        frames_to_insert
    )
    conn.commit()
    conn.close()


def update_extracted_frame_count(video_uuid, count):
    conn = get_db_connection()
    conn.execute('UPDATE videos SET extracted_frame_count = ? WHERE video_uuid = ?', (count, video_uuid))
    conn.commit()
    conn.close()


def delete_video(video_uuid):
    conn = get_db_connection()
    conn.execute('DELETE FROM videos WHERE video_uuid = ?', (video_uuid,))
    conn.commit()
    conn.close()


def get_video_frames(video_uuid):
    conn = get_db_connection()
    frames = conn.execute('SELECT * FROM video_frames WHERE video_uuid = ? ORDER BY frame_number ASC',
                          (video_uuid,)).fetchall()
    conn.close()
    return [dict(row) for row in frames]


def save_frame_bboxes(video_uuid, frame_number, bboxes_text):
    conn = get_db_connection()
    cursor = conn.cursor()
    frame = cursor.execute(
        'SELECT frame_id FROM video_frames WHERE video_uuid = ? AND frame_number = ?',
        (video_uuid, frame_number)
    ).fetchone()

    if not frame:
        conn.close()
        logging.error(f"无法为 video {video_uuid}, frame {frame_number} 找到 frame_id。")
        return

    frame_id = frame['frame_id']
    cursor.execute(
        'UPDATE video_frames SET bboxes_text = ?, suggested_bboxes_text = ? WHERE frame_id = ?',
        (bboxes_text, '', frame_id)
    )
    # 删除旧标签重新计算
    cursor.execute('DELETE FROM frame_labels WHERE frame_id = ?', (frame_id,))
    unique_labels_in_frame = set(extract_labels(bboxes_text))
    if unique_labels_in_frame:
        labels_to_insert = [(frame_id, label) for label in unique_labels_in_frame]
        cursor.executemany(
            'INSERT INTO frame_labels (frame_id, label_name) VALUES (?, ?)',
            labels_to_insert
        )

    # 更新已标注计数
    new_labeled_count = cursor.execute(
        "SELECT COUNT(*) FROM video_frames WHERE video_uuid = ? AND ((bboxes_text IS NOT NULL AND bboxes_text != '') OR (tags IS NOT NULL AND tags != '[]' AND tags != ''))",
        (video_uuid,)
    ).fetchone()[0]

    cursor.execute(
        'UPDATE videos SET labeled_frame_count = ? WHERE video_uuid = ?',
        (new_labeled_count, video_uuid)
    )

    conn.commit()
    conn.close()


def save_frame_suggestions(video_uuid, frame_number, suggested_bboxes_text):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE video_frames SET suggested_bboxes_text = ? WHERE video_uuid = ? AND frame_number = ?',
        (suggested_bboxes_text, video_uuid, frame_number)
    )
    conn.commit()
    conn.close()


def save_frame_tags(video_uuid, frame_number, tags_json_string):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE video_frames SET tags = ? WHERE video_uuid = ? AND frame_number = ?',
        (tags_json_string, video_uuid, frame_number)
    )
    new_labeled_count = cursor.execute(
        "SELECT COUNT(*) FROM video_frames WHERE video_uuid = ? AND ((bboxes_text IS NOT NULL AND bboxes_text != '') OR (tags IS NOT NULL AND tags != '[]' AND tags != ''))",
        (video_uuid,)
    ).fetchone()[0]
    cursor.execute(
        'UPDATE videos SET labeled_frame_count = ? WHERE video_uuid = ?',
        (new_labeled_count, video_uuid)
    )
    conn.commit()
    conn.close()


def get_next_safe_frame_number(video_uuid):
    """
    获取下一个安全的帧序号。
    逻辑：查找当前最大的 frame_number，然后 +1。
    如果没有任何帧，则返回 0。
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    result = cursor.execute('SELECT MAX(frame_number) FROM video_frames WHERE video_uuid = ?', (video_uuid,)).fetchone()
    conn.close()

    max_val = result[0]
    if max_val is None:
        return 0
    return max_val + 1


def add_frames_to_video(video_uuid, frames_data_list):
    """
    通用函数：将准备好的帧数据（图像二进制）批量写入磁盘和数据库。
    frames_data_list: list of bytes (图片二进制数据)
    """
    if not frames_data_list:
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1. 获取安全的起始序号，防止覆盖
        start_frame_number = get_next_safe_frame_number(video_uuid)

        db_rows_to_insert = []

        # 2. 循环保存文件并准备 DB 数据
        for i, image_bytes in enumerate(frames_data_list):
            new_frame_number = start_frame_number + i

            # 保存到磁盘 (调用 file_storage)
            file_storage.save_frame_image(video_uuid, new_frame_number, image_bytes)

            # 准备 DB 记录: (uuid, frame_num, bbox, suggest, tags, include)
            db_rows_to_insert.append((video_uuid, new_frame_number, '', '', '', 1))

        # 3. 批量插入数据库
        if db_rows_to_insert:
            cursor.executemany(
                'INSERT INTO video_frames (video_uuid, frame_number, bboxes_text, suggested_bboxes_text, tags, include_frame_in_dataset) VALUES (?, ?, ?, ?, ?, ?)',
                db_rows_to_insert
            )

        # 4. 更新视频主表统计信息
        # 重新计算总帧数
        final_count = \
        cursor.execute('SELECT COUNT(*) FROM video_frames WHERE video_uuid = ?', (video_uuid,)).fetchone()[0]

        cursor.execute(
            'UPDATE videos SET frame_count = ?, extracted_frame_count = ?, included_frame_count = ? WHERE video_uuid = ?',
            (final_count, final_count, final_count, video_uuid)
        )

        conn.commit()
        return len(frames_data_list)

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def add_frames_from_upload(video_uuid, frame_files):
    """
    兼容旧逻辑的入口函数。
    读取 Flask FileStorage 对象列表，转换为 bytes 列表，然后调用通用函数。
    """
    frames_data = [f.read() for f in frame_files]
    return add_frames_to_video(video_uuid, frames_data)


def create_annotation_task(video_uuid, assigned_to, description, start_frame, end_frame):
    conn = get_db_connection()
    cursor = conn.cursor()
    existing_tasks = cursor.execute(
        'SELECT start_frame, end_frame FROM annotation_tasks WHERE video_uuid = ?', (video_uuid,)
    ).fetchall()
    for task in existing_tasks:
        if start_frame <= task['end_frame'] and end_frame >= task['start_frame']:
            conn.close()
            raise ValueError(f"Frame range overlaps with an existing task ({task['start_frame']}-{task['end_frame']}).")
    task_uuid = str(uuid.uuid4().hex)
    create_time_ms = int(time.time() * 1000)
    cursor.execute(
        '''INSERT INTO annotation_tasks
           (task_uuid, video_uuid, assigned_to, description, start_frame, end_frame, status, create_time_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (task_uuid, video_uuid, assigned_to, description, start_frame, end_frame, 'PENDING', create_time_ms)
    )
    conn.commit()
    conn.close()
    return task_uuid


def get_tasks_for_video(video_uuid):
    conn = get_db_connection()
    tasks = conn.execute('SELECT * FROM annotation_tasks WHERE video_uuid = ? ORDER BY create_time_ms DESC',
                         (video_uuid,)).fetchall()
    conn.close()
    return [dict(row) for row in tasks]


def get_task_entity(task_uuid):
    conn = get_db_connection()
    task = conn.execute('SELECT * FROM annotation_tasks WHERE task_uuid = ?', (task_uuid,)).fetchone()
    conn.close()
    return dict(task) if task else None


def delete_task(task_uuid):
    conn = get_db_connection()
    conn.execute('DELETE FROM annotation_tasks WHERE task_uuid = ?', (task_uuid,))
    conn.commit()
    conn.close()


def update_task_status(task_uuid, status):
    conn = get_db_connection()
    conn.execute('UPDATE annotation_tasks SET status = ? WHERE task_uuid = ?', (status, task_uuid))
    conn.commit()
    conn.close()


def add_class_label(label_name):
    conn = get_db_connection()
    try:
        create_time_ms = int(time.time() * 1000)
        conn.execute('INSERT INTO class_labels (label_name, create_time_ms) VALUES (?, ?)',
                     (label_name, create_time_ms))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()


def get_all_class_labels():
    conn = get_db_connection()
    labels = conn.execute('SELECT label_name FROM class_labels ORDER BY label_name ASC').fetchall()
    conn.close()
    return [row['label_name'] for row in labels]


def delete_class_label(label_name):
    conn = get_db_connection()
    conn.execute('DELETE FROM class_labels WHERE label_name = ?', (label_name,))
    conn.commit()
    conn.close()


def get_all_frames_with_class(class_name):
    conn = get_db_connection()
    query = """
            SELECT T1.*, T2.width, T2.height
            FROM video_frames AS T1
                     INNER JOIN videos AS T2 ON T1.video_uuid = T2.video_uuid
                     INNER JOIN frame_labels AS T3 ON T1.frame_id = T3.frame_id
            WHERE T3.label_name = ? \
            """
    frames = conn.execute(query, (class_name,)).fetchall()
    conn.close()

    return [dict(row) for row in frames]


def add_class_tag(tag_name):
    conn = get_db_connection()
    try:
        create_time_ms = int(time.time() * 1000)
        conn.execute('INSERT INTO class_tags (tag_name, create_time_ms) VALUES (?, ?)', (tag_name, create_time_ms))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()


def get_all_class_tags():
    conn = get_db_connection()
    tags = conn.execute('SELECT tag_name FROM class_tags ORDER BY tag_name ASC').fetchall()
    conn.close()
    return [row['tag_name'] for row in tags]


def delete_class_tag(tag_name):
    conn = get_db_connection()
    conn.execute('DELETE FROM class_tags WHERE tag_name = ?', (tag_name,))
    conn.commit()
    conn.close()


def create_dataset_entry(description, video_uuids, create_time_ms, eval_percent, test_percent):
    conn = get_db_connection()
    dataset_uuid = str(uuid.uuid4().hex)
    conn.execute(
        'INSERT INTO datasets (dataset_uuid, description, video_uuids, create_time_ms, status, eval_percent, test_percent) VALUES (?, ?, ?, ?, ?, ?, ?)',
        (dataset_uuid, description, json.dumps(video_uuids), create_time_ms, 'PENDING', eval_percent, test_percent)
    )
    conn.commit()
    conn.close()
    return dataset_uuid


def update_dataset_status(dataset_uuid, status, message="", zip_path="", sorted_label_list=None):
    conn = get_db_connection()
    if sorted_label_list is not None:
        conn.execute(
            'UPDATE datasets SET status=?, status_message=?, zip_path=?, sorted_label_list=? WHERE dataset_uuid=?',
            (status, message, zip_path, json.dumps(sorted_label_list), dataset_uuid)
        )
    else:
        conn.execute(
            'UPDATE datasets SET status=?, status_message=?, zip_path=? WHERE dataset_uuid=?',
            (status, message, zip_path, dataset_uuid)
        )
    conn.commit()
    conn.close()


def get_dataset_list():
    conn = get_db_connection()
    datasets = conn.execute('SELECT * FROM datasets ORDER BY create_time_ms DESC').fetchall()
    conn.close()
    return [dict(row) for row in datasets]


def get_dataset_entity(dataset_uuid):
    conn = get_db_connection()
    dataset = conn.execute('SELECT * FROM datasets WHERE dataset_uuid = ?', (dataset_uuid,)).fetchone()
    conn.close()
    return dict(dataset) if dataset else None


def delete_dataset(dataset_uuid):
    conn = get_db_connection()
    conn.execute('DELETE FROM datasets WHERE dataset_uuid = ?', (dataset_uuid,))
    conn.commit()
    conn.close()


def import_model_metadata(description, label_filename, model_type, create_time_ms):
    conn = get_db_connection()
    model_uuid = str(uuid.uuid4().hex)
    conn.execute(
        'INSERT INTO models (model_uuid, description, label_filename, model_type, create_time_ms) VALUES (?, ?, ?, ?, ?)',
        (model_uuid, description, label_filename, model_type, create_time_ms)
    )
    conn.commit()
    conn.close()
    return model_uuid


def get_model_list():
    conn = get_db_connection()
    models = conn.execute('SELECT * FROM models ORDER BY create_time_ms DESC').fetchall()
    conn.close()
    return [dict(row) for row in models]


def get_model_entity(model_uuid):
    conn = get_db_connection()
    model = conn.execute('SELECT * FROM models WHERE model_uuid = ?', (model_uuid,)).fetchone()
    conn.close()
    return dict(model) if model else None


def delete_model(model_uuid):
    conn = get_db_connection()
    conn.execute('DELETE FROM models WHERE model_uuid = ?', (model_uuid,))
    conn.commit()
    conn.close()


def get_frame_numbers_for_video(video_uuid):
    conn = get_db_connection()
    frames = conn.execute('SELECT frame_number FROM video_frames WHERE video_uuid = ? ORDER BY frame_number ASC',
                          (video_uuid,)).fetchall()
    conn.close()
    return [row['frame_number'] for row in frames]