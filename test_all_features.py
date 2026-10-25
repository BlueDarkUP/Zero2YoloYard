import requests
import json
import os
import cv2
import numpy as np
import time
import traceback

BASE_URL = "http://127.0.0.1:5000"

def run_comprehensive_audit():
    print("=" * 70)
    print("STARTING FULL REPOSITORY FUNCTIONALITY AUDIT")
    print("=" * 70)
    
    passed = 0
    failed = 0
    failures = []

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [FAIL] {name}: {e}\n{tb}")
            failed += 1
            failures.append((name, str(e)))

    # 1. Main views
    def test_root():
        res = requests.get(f"{BASE_URL}/")
        assert res.status_code == 200, f"Status: {res.status_code}"
        assert "Zero2Yolo" in res.text or "TERMINAL" in res.text or "TELEMETRY" in res.text
    test("GET / (Root Dashboard View)", test_root)

    def test_settings_api():
        res = requests.get(f"{BASE_URL}/api/settings")
        assert res.status_code == 200
        data = res.json()
        assert data.get("success") is True
        assert "settings" in data
        assert "sam_model_checkpoint" in data["settings"]
    test("GET /api/settings (Settings Retrieval)", test_settings_api)

    def test_detect_hardware():
        res = requests.get(f"{BASE_URL}/api/detect_hardware")
        assert res.status_code == 200
        data = res.json()
        assert "cpu_cores" in data
        assert "ram_gb" in data
        assert "has_cuda" in data
    test("GET /api/detect_hardware (Hardware Telemetry)", test_detect_hardware)

    def test_export_formats():
        res = requests.get(f"{BASE_URL}/api/export_formats")
        assert res.status_code == 200
        data = res.json()
        assert data.get("success") is True
        assert len(data.get("formats", [])) > 0
    test("GET /api/export_formats (Supported Export Formats)", test_export_formats)

    def test_list_videos():
        res = requests.get(f"{BASE_URL}/listVideos")
        assert res.status_code == 200
        data = res.json()
        assert "all_videos" in data
    test("GET /listVideos (Video Stream Catalog)", test_list_videos)

    def test_list_datasets():
        res = requests.get(f"{BASE_URL}/listDatasets")
        assert res.status_code == 200
        data = res.json()
        assert "datasets" in data
    test("GET /listDatasets (Dataset Repository)", test_list_datasets)

    def test_list_models():
        res = requests.get(f"{BASE_URL}/listModels")
        assert res.status_code == 200
        data = res.json()
        assert "models" in data
    test("GET /listModels (Model Engine Catalog)", test_list_models)

    def test_list_classes():
        res = requests.get(f"{BASE_URL}/listClasses")
        assert res.status_code == 200
        data = res.json()
        assert data.get("success") is True
        assert "labels" in data
    test("GET /listClasses (Class Label Registry)", test_list_classes)

    def test_export_settings():
        res = requests.get(f"{BASE_URL}/api/settings/export")
        assert res.status_code == 200
        assert "attachment" in res.headers.get("Content-Disposition", "")
        parsed = res.json()
        assert "sam_model_checkpoint" in parsed
    test("GET /api/settings/export (Settings JSON Export)", test_export_settings)

    def test_import_settings():
        settings_to_import = {"max_cache_size": 42}
        res = requests.post(f"{BASE_URL}/api/settings/import", json=settings_to_import)
        assert res.status_code == 200
        assert res.json().get("success") is True
        verify_res = requests.get(f"{BASE_URL}/api/settings").json()
        assert verify_res["settings"]["max_cache_size"] == 42
    test("POST /api/settings/import (Settings JSON Import)", test_import_settings)

    def test_save_settings_endpoint():
        res = requests.post(f"{BASE_URL}/api/settings", json={"max_cache_size": 30})
        assert res.status_code == 200
        assert res.json().get("success") is True
    test("POST /api/settings (Settings Update Endpoint)", test_save_settings_endpoint)

    def test_clear_cache():
        res = requests.post(f"{BASE_URL}/api/clear_cache")
        assert res.status_code == 200
        assert res.json().get("success") is True
    test("POST /api/clear_cache (Smart Select Cache Purge)", test_clear_cache)

    # Test Video Upload & Frame Extraction & Frame Retrieval
    test_video_uuid = None
    def test_video_flow():
        nonlocal test_video_uuid
        temp_mp4 = "_audit_temp_vid.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_mp4, fourcc, 10.0, (64, 64))
        for i in range(10):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:] = (i * 25, 120, 200)
            out.write(frame)
        out.release()

        with open(temp_mp4, "rb") as f:
            res = requests.post(f"{BASE_URL}/uploadVideo", files={"video_file": (temp_mp4, f, "video/mp4")}, data={
                "description": f"_AUDIT_STREAM_{int(time.time())}",
                "frame_interval": "1",
                "annotation_type": "detection"
            })
        if os.path.exists(temp_mp4):
            os.remove(temp_mp4)

        assert res.status_code == 200
        data = res.json()
        assert data.get("success") is True
        test_video_uuid = data.get("video_uuid")
        assert test_video_uuid is not None

        # Wait up to 5 seconds for status READY
        for _ in range(10):
            time.sleep(0.5)
            v_list = requests.get(f"{BASE_URL}/listVideos").json().get("all_videos", [])
            matched = [v for v in v_list if v.get("video_uuid") == test_video_uuid]
            if matched and matched[0].get("status") == "READY":
                break
    test("POST /uploadVideo & Stream Ingestion", test_video_flow)

    def test_label_video_view():
        res = requests.get(f"{BASE_URL}/labelVideo?video_uuid={test_video_uuid}")
        assert res.status_code == 200
        assert "canvas-container" in res.text or "labelVideo" in res.text or "ide-layout" in res.text
    test("GET /labelVideo (Annotation Studio IDE View)", test_label_video_view)

    def test_task_management_flow():
        # List tasks (default task was auto-created)
        list_res = requests.get(f"{BASE_URL}/listTasks?video_uuid={test_video_uuid}")
        assert list_res.status_code == 200
        tasks = list_res.json().get("tasks", [])
        assert len(tasks) > 0, "Expected default task to exist"
        
        # Delete default task
        first_task_uuid = tasks[0]["task_uuid"]
        del_res = requests.post(f"{BASE_URL}/deleteTask", json={"task_uuid": first_task_uuid})
        assert del_res.status_code == 200
        assert del_res.json().get("success") is True

        # Create new custom task
        res = requests.post(f"{BASE_URL}/createTask", json={
            "video_uuid": test_video_uuid,
            "assigned_to": "AUDIT_OPERATOR",
            "start_frame": 0,
            "end_frame": 4,
            "description": "Custom Task"
        })
        assert res.status_code == 200, f"Status: {res.status_code}, Body: {res.text}"
        data = res.json()
        assert data.get("success") is True, f"Body: {data}"
        task_uuid = data.get("task_uuid")

        # Verify new task is listed
        list_res2 = requests.get(f"{BASE_URL}/listTasks?video_uuid={test_video_uuid}")
        assert any(t.get("task_uuid") == task_uuid for t in list_res2.json().get("tasks", []))
    test("POST /createTask, GET /listTasks, POST /deleteTask", test_task_management_flow)

    def test_cleanup_video():
        if test_video_uuid:
            res = requests.post(f"{BASE_URL}/deleteVideo", json={"video_uuid": test_video_uuid})
            assert res.status_code == 200
            assert res.json().get("success") is True
    test("POST /deleteVideo (Video Stream Purge)", test_cleanup_video)

    def test_setup_wizard_view():
        res = requests.get(f"{BASE_URL}/setup")
        assert res.status_code == 200
        assert "INITIALIZATION" in res.text or "wizard" in res.text or "setup" in res.text.lower()
    test("GET /setup (First-Run Configuration Stepper)", test_setup_wizard_view)

    print("\n" + "=" * 70)
    print(f"AUDIT COMPLETE: {passed} PASSED, {failed} FAILED")
    print("=" * 70)
    return failed == 0

if __name__ == "__main__":
    success = run_comprehensive_audit()
    exit(0 if success else 1)
