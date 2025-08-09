import json
import sqlite3
import time
import uuid
import config
import file_storage


def get_db_connection():
    conn = sqlite3.connect(config.DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        video_uuid TEXT PRIMARY KEY,
        description TEXT NOT NULL UNIQUE,
        video_filename TEXT,
        file_size INTEGER,
        create_time_ms INTEGER,
        status TEXT, -- 'UPLOADING', 'EXTRACTING', 'READY', 'FAILED'
        status_message TEXT,
        width INTEGER,
        height INTEGER,
        fps REAL,
        frame_count INTEGER,
        extracted_frame_count INTEGER DEFAULT 0,
        included_frame_count INTEGER DEFAULT 0,
        labeled_frame_count INTEGER DEFAULT 0
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS video_frames (
        frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_uuid TEXT,
        frame_number INTEGER,
        bboxes_text TEXT,
        include_frame_in_dataset INTEGER, -- 0 for false, 1 for true
        FOREIGN KEY (video_uuid) REFERENCES videos (video_uuid) ON DELETE CASCADE
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_uuid TEXT PRIMARY KEY,
        description TEXT NOT NULL UNIQUE,
        video_uuids TEXT, -- JSON list of video_uuids
        create_time_ms INTEGER,
        status TEXT, -- 'PENDING', 'PROCESSING', 'READY', 'FAILED'
        status_message TEXT,
        zip_path TEXT,
        sorted_label_list TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS models (
        model_uuid TEXT PRIMARY KEY,
        description TEXT NOT NULL UNIQUE,
        create_time_ms INTEGER
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS annotation_tasks (
        task_uuid TEXT PRIMARY KEY,
        video_uuid TEXT NOT NULL,
        assigned_to TEXT NOT NULL,      -- 任务分配给谁 (简单文本)
        description TEXT,               -- 任务描述
        start_frame INTEGER NOT NULL,
        end_frame INTEGER NOT NULL,
        status TEXT,                    -- 'PENDING', 'IN_PROGRESS', 'COMPLETED'
        create_time_ms INTEGER,
        FOREIGN KEY (video_uuid) REFERENCES videos (video_uuid) ON DELETE CASCADE
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS class_labels (
        label_id INTEGER PRIMARY KEY AUTOINCREMENT,
        label_name TEXT NOT NULL UNIQUE,
        create_time_ms INTEGER
    )
    ''')

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
    videos = conn.execute(
        'SELECT * FROM videos WHERE status = "READY" AND labeled_frame_count > 0 ORDER BY create_time_ms DESC').fetchall()
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

def update_video_after_extraction_start(video_uuid, width, height, fps, frame_count):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE videos SET width=?, height=?, fps=?, frame_count=?, included_frame_count=?, status=? WHERE video_uuid=?',
        (width, height, fps, frame_count, frame_count, 'EXTRACTING', video_uuid)
    )
    frames_to_insert = [(video_uuid, i, '', 1) for i in range(frame_count)]
    cursor.executemany(
        'INSERT INTO video_frames (video_uuid, frame_number, bboxes_text, include_frame_in_dataset) VALUES (?, ?, ?, ?)',
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
    cursor.execute(
        'UPDATE video_frames SET bboxes_text = ? WHERE video_uuid = ? AND frame_number = ?',
        (bboxes_text, video_uuid, frame_number)
    )
    new_labeled_count = cursor.execute(
        "SELECT COUNT(*) FROM video_frames WHERE video_uuid = ? AND bboxes_text IS NOT NULL AND bboxes_text != ''",
        (video_uuid,)
    ).fetchone()[0]
    cursor.execute(
        'UPDATE videos SET labeled_frame_count = ? WHERE video_uuid = ?',
        (new_labeled_count, video_uuid)
    )
    conn.commit()
    conn.close()

def add_frames_from_upload(video_uuid, frame_files):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        video_data = cursor.execute('SELECT frame_count FROM videos WHERE video_uuid = ?', (video_uuid,)).fetchone()
        if not video_data:
            raise ValueError("Video UUID not found in database.")
        start_frame_number = video_data['frame_count']
        frames_to_insert = []
        for i, file_storage_obj in enumerate(frame_files):
            new_frame_number = start_frame_number + i
            image_bytes = file_storage_obj.read()
            file_storage.save_frame_image(video_uuid, new_frame_number, image_bytes)
            frames_to_insert.append((video_uuid, new_frame_number, '', 1))
        if frames_to_insert:
            cursor.executemany(
                'INSERT INTO video_frames (video_uuid, frame_number, bboxes_text, include_frame_in_dataset) VALUES (?, ?, ?, ?)',
                frames_to_insert
            )
        new_total_frames = start_frame_number + len(frames_to_insert)
        cursor.execute(
            'UPDATE videos SET frame_count = ?, extracted_frame_count = ?, included_frame_count = ? WHERE video_uuid = ?',
            (new_total_frames, new_total_frames, new_total_frames, video_uuid)
        )
        conn.commit()
        return len(frames_to_insert)
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


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
        conn.execute('INSERT INTO class_labels (label_name, create_time_ms) VALUES (?, ?)', (label_name, create_time_ms))
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


def create_dataset_entry(description, video_uuids, create_time_ms):
    conn = get_db_connection()
    dataset_uuid = str(uuid.uuid4().hex)
    conn.execute(
        'INSERT INTO datasets (dataset_uuid, description, video_uuids, create_time_ms, status) VALUES (?, ?, ?, ?, ?)',
        (dataset_uuid, description, json.dumps(video_uuids), create_time_ms, 'PENDING')
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


def import_model_metadata(description, create_time_ms):
    conn = get_db_connection()
    model_uuid = str(uuid.uuid4().hex)
    conn.execute(
        'INSERT INTO models (model_uuid, description, create_time_ms) VALUES (?, ?, ?)',
        (model_uuid, description, create_time_ms)
    )
    conn.commit()
    conn.close()
    return model_uuid

def get_model_list():
    conn = get_db_connection()
    models = conn.execute('SELECT * FROM models ORDER BY create_time_ms DESC').fetchall()
    conn.close()
    return [dict(row) for row in models]

def delete_model(model_uuid):
    conn = get_db_connection()
    conn.execute('DELETE FROM models WHERE model_uuid = ?', (model_uuid,))
    conn.commit()
    conn.close()