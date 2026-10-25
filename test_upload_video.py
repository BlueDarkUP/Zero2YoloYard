import os
import cv2
import numpy as np
import requests
import time

def test_video_upload():
    # Clean up previous test videos if any
    for prev_uuid in ['ffe36f12f29449969b13294e2ad454ec', '04a2c6e041ad4224ae5994e9de6444b1']:
        requests.post("http://127.0.0.1:5000/deleteVideo", json={"video_uuid": prev_uuid})

    # 1. Create a tiny test MP4
    temp_mp4 = "test_upload_sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_mp4, fourcc, 10.0, (64, 64))
    for i in range(10):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:] = (i * 20, 100, 200)
        out.write(frame)
    out.release()
    print(f"Generated {temp_mp4}, size: {os.path.getsize(temp_mp4)} bytes")

    # 2. Upload to http://127.0.0.1:5000/uploadVideo
    url = "http://127.0.0.1:5000/uploadVideo"
    desc = f"TEST_STREAM_{int(time.time())}"
    
    with open(temp_mp4, "rb") as f:
        files = {
            "video_file": (temp_mp4, f, "video/mp4")
        }
        data = {
            "description": desc,
            "frame_interval": "2",
            "annotation_type": "detection"
        }
        res = requests.post(url, files=files, data=data)
        
    print("Status code:", res.status_code)
    print("Response JSON:", res.json())
    
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    result = res.json()
    assert result.get("success") is True, "Expected success to be True"
    video_uuid = result.get("video_uuid")
    print(f"Video uploaded successfully with UUID: {video_uuid}")

    # 3. Poll /listVideos to verify video status
    time.sleep(2)
    list_res = requests.get("http://127.0.0.1:5000/listVideos")
    videos = list_res.json().get("all_videos", [])
    matched = [v for v in videos if v.get("video_uuid") == video_uuid]
    assert len(matched) == 1, f"Uploaded video not found in /listVideos! matched: {matched}"
    print(f"Found uploaded video in list: {matched[0]['description']} (status: {matched[0].get('status')})")

    # 4. Clean up video entry via /deleteVideo
    del_res = requests.post("http://127.0.0.1:5000/deleteVideo", json={"video_uuid": video_uuid})
    print("Delete video response:", del_res.json())
    assert del_res.json().get("success") is True

    # 5. Clean up local temp file
    if os.path.exists(temp_mp4):
        os.remove(temp_mp4)

    print("\n>>> ALL VIDEO UPLOAD CHECKS PASSED 100%! <<<")

if __name__ == "__main__":
    test_video_upload()
