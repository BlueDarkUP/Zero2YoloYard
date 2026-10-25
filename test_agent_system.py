"""
Zero2YoloYard - Agent System End-to-End Verification Suite
==========================================================
Comprehensive automated testing of the AI Agent Hub, REST endpoints,
CLI command dispatcher, tool schemas, and autonomous labeling execution.
"""

import os
import sys
import json
import time
import subprocess
import urllib.request
import urllib.error
import agent_hub

BASE_URL = "http://127.0.0.1:5000"


def print_step(title):
    print(f"\n{'='*70}\n[TEST] {title}\n{'='*70}")


def assert_true(cond, msg):
    if not cond:
        print(f"  [FAIL] {msg}")
        raise AssertionError(msg)
    print(f"  [PASS] {msg}")


def test_registry_integrity():
    print_step("Agent Tool Registry & Schema Verification")
    tools = agent_hub.TOOL_REGISTRY
    assert_true(len(tools) >= 65, f"Registered tool count: {len(tools)} >= 65")

    # Verify absence of datasetAnalysis / QC review tools per explicit requirement
    for tool_name in tools.keys():
        assert_true("dataset_analysis" not in tool_name.lower(), f"No dataset analysis in {tool_name}")
        assert_true("consistency_check" not in tool_name.lower(), f"No consistency check in {tool_name}")
        assert_true("qc_review" not in tool_name.lower(), f"No QC review in {tool_name}")

    # Verify all tools have required schema metadata
    for name, tool in tools.items():
        assert_true("category" in tool, f"{name} has category")
        assert_true("description" in tool and len(tool["description"]) > 10, f"{name} has descriptive text")
        assert_true("parameters" in tool and "type" in tool["parameters"], f"{name} has parameters schema")
        assert_true("handler" in tool and callable(tool["handler"]), f"{name} has callable handler")

    schemas = agent_hub.get_all_tools_schema()
    assert_true(len(schemas) == len(tools), f"OpenAPI schemas count matches registry ({len(schemas)})")


def test_agent_rest_endpoints():
    print_step("Agent REST Endpoints via HTTP (Waitress Backend)")
    
    # 1. GET /api/agent/tools
    req = urllib.request.Request(f"{BASE_URL}/api/agent/tools")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert_true(resp.status == 200, "GET /api/agent/tools returns HTTP 200")
        data = json.loads(resp.read().decode())
        assert_true(data.get("success") is True, "/api/agent/tools success is True")
        assert_true(data.get("count") >= 65, f"/api/agent/tools count is {data.get('count')} (>= 65)")

    # 2. GET /api/agent/prompt
    req = urllib.request.Request(f"{BASE_URL}/api/agent/prompt")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert_true(resp.status == 200, "GET /api/agent/prompt returns HTTP 200")
        prompt_text = resp.read().decode()
        assert_true("AUTONOMOUS AI AGENT OPERATIONAL SPECIFICATION" in prompt_text, "Prompt contains specification header")
        assert_true("Playbook 1" in prompt_text, "Prompt contains autonomous playbooks")

    # 3. POST /api/agent/execute - System Status
    payload = json.dumps({"tool": "z2yy_get_system_status"}).encode()
    req = urllib.request.Request(f"{BASE_URL}/api/agent/execute", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert_true(resp.status == 200, "POST /api/agent/execute (status) returns HTTP 200")
        res = json.loads(resp.read().decode())
        assert_true(res.get("success") is True, "System status execution success")
        assert_true("cuda_available" in res["result"], "System status contains cuda_available")

    # 4. POST /api/agent/execute - Hardware Telemetry
    payload = json.dumps({"tool": "z2yy_detect_hardware"}).encode()
    req = urllib.request.Request(f"{BASE_URL}/api/agent/execute", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert_true(resp.status == 200, "POST /api/agent/execute (hardware) returns HTTP 200")
        res = json.loads(resp.read().decode())
        assert_true(res.get("success") is True, "Hardware detection success")
        assert_true("recommended_preset" in res["result"], "Hardware detection contains recommended_preset")

    # 5. POST /api/agent/execute - Error Handling (missing required arg)
    payload = json.dumps({"tool": "z2yy_set_class_prompt", "arguments": {"class_name": "only_name"}}).encode()
    req = urllib.request.Request(f"{BASE_URL}/api/agent/execute", data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        assert_true(False, "Should have rejected missing required argument")
    except urllib.error.HTTPError as he:
        assert_true(he.code == 400, "HTTP 400 returned for missing required parameter")
        err_data = json.loads(he.read().decode())
        assert_true(err_data.get("success") is False, "success is False on error")

    # 6. POST /api/agent/execute - Error Handling (unknown tool)
    payload = json.dumps({"tool": "non_existent_tool_xyz"}).encode()
    req = urllib.request.Request(f"{BASE_URL}/api/agent/execute", data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        assert_true(False, "Should have rejected unknown tool")
    except urllib.error.HTTPError as he:
        assert_true(he.code == 400, "HTTP 400 returned for unknown tool")


def test_agent_cli_dispatcher():
    print_step("Agent CLI Dispatcher (Standalone & EXE compatibility)")
    python_exe = sys.executable

    # 1. list-tools command
    p = subprocess.run([python_exe, "app.py", "agent", "list-tools"], capture_output=True, text=True)
    assert_true(p.returncode == 0, "CLI: app.py agent list-tools returns 0")
    tools = json.loads(p.stdout[p.stdout.find("["):])
    assert_true(len(tools) >= 65, f"CLI output parsed {len(tools)} tools (>= 65)")

    # 2. describe command
    p = subprocess.run([python_exe, "app.py", "agent", "describe", "z2yy_upload_video"], capture_output=True, text=True)
    assert_true(p.returncode == 0, "CLI: app.py agent describe returns 0")
    desc_data = json.loads(p.stdout[p.stdout.find("{"):])
    assert_true(desc_data.get("name") == "z2yy_upload_video", "CLI describe matched tool name")

    # 3. execute command
    p = subprocess.run([python_exe, "app.py", "agent", "execute", "z2yy_detect_hardware"], capture_output=True, text=True)
    assert_true(p.returncode == 0, "CLI: app.py agent execute z2yy_detect_hardware returns 0")
    exec_data = json.loads(p.stdout[p.stdout.find("{"):])
    assert_true(exec_data.get("success") is True, "CLI execution returned success: true")
    assert_true(exec_data.get("tool") == "z2yy_detect_hardware", "CLI execution returned correct tool")

    # 4. dump-prompt command
    p = subprocess.run([python_exe, "app.py", "agent", "dump-prompt"], capture_output=True, text=True)
    assert_true(p.returncode == 0, "CLI: app.py agent dump-prompt returns 0")
    assert_true("ZERO2YOLOYARD" in p.stdout, "CLI dump-prompt outputs Markdown documentation")

    # 5. dump-schema command
    p = subprocess.run([python_exe, "app.py", "agent", "dump-schema"], capture_output=True, text=True)
    assert_true(p.returncode == 0, "CLI: app.py agent dump-schema returns 0")
    schemas = json.loads(p.stdout[p.stdout.find("["):])
    assert_true(len(schemas) >= 65, f"CLI dump-schema returns JSON schema list ({len(schemas)} >= 65)")


def test_autonomous_workflow_lifecycle():
    print_step("Autonomous End-to-End Workflow & Granular Human Control Parity")
    
    ts = int(time.time())
    test_class = f"agent_cls_{ts}"
    renamed_class = f"agent_renamed_{ts}"
    test_prompt = "a special robotic test apparatus component"

    # Step 1: Add class
    r1 = agent_hub.execute_tool("z2yy_add_class", {"class_name": test_class})
    assert_true(r1.get("success") is True, f"z2yy_add_class created {test_class}")

    # Step 2: Check class usage
    r_usage = agent_hub.execute_tool("z2yy_check_class_usage", {"class_name": test_class})
    assert_true(r_usage.get("success") is True, "z2yy_check_class_usage executed")
    assert_true(r_usage.get("result", {}).get("usage_count") == 0, "Initial usage count is 0")

    # Step 3: Rename class
    r_rename = agent_hub.execute_tool("z2yy_rename_class", {
        "old_class_name": test_class,
        "new_class_name": renamed_class
    })
    assert_true(r_rename.get("success") is True, f"z2yy_rename_class renamed to {renamed_class}")

    # Step 4: Set class prompt
    r2 = agent_hub.execute_tool("z2yy_set_class_prompt", {"class_name": renamed_class, "sam3_prompt": test_prompt})
    assert_true(r2.get("success") is True, f"z2yy_set_class_prompt assigned prompt")

    # Step 5: Verify class & prompt in catalog
    r3 = agent_hub.execute_tool("z2yy_list_class_prompts")
    classes = r3.get("result", {}).get("classes", [])
    matching = [c for c in classes if c.get("class_name") == renamed_class]
    assert_true(len(matching) == 1, f"Found {renamed_class} in class prompts")
    assert_true(matching[0].get("sam3_prompt") == test_prompt, "Class prompt matches exactly")

    # Step 6: Pose schema definition
    r4 = agent_hub.execute_tool("z2yy_save_pose_schema", {
        "class_name": renamed_class,
        "keypoints": ["head", "center", "tail"],
        "skeleton": [[0, 1], [1, 2]]
    })
    assert_true(r4.get("success") is True, "z2yy_save_pose_schema saved keypoints and skeleton")

    r5 = agent_hub.execute_tool("z2yy_get_pose_schemas")
    schemas = r5.get("result", {}).get("schemas", {})
    assert_true(renamed_class in schemas, f"{renamed_class} schema retrieved from database")

    # Step 7: System Settings & Hardware Telemetry
    r_hw = agent_hub.execute_tool("z2yy_detect_hardware")
    assert_true(r_hw.get("success") is True, "z2yy_detect_hardware succeeded")

    r_exp = agent_hub.execute_tool("z2yy_export_settings")
    assert_true(r_exp.get("success") is True, "z2yy_export_settings returned settings")

    r_cache = agent_hub.execute_tool("z2yy_clear_cache")
    assert_true(r_cache.get("success") is True, "z2yy_clear_cache succeeded")

    # Step 8: Export formats catalog
    r6 = agent_hub.execute_tool("z2yy_get_export_formats")
    formats = r6.get("result", {}).get("formats", [])
    assert_true(len(formats) > 5, f"Supported export formats: {len(formats)} > 5")
    assert_true(any(f.get("key") == "yolo_v8_detect" or f.get("format_id") == "yolo_v8_detect" for f in formats), "yolo_v8_detect supported")

    # Step 9: Dataset listing
    r7 = agent_hub.execute_tool("z2yy_list_datasets")
    assert_true(r7.get("success") is True, "z2yy_list_datasets executed successfully")

    # Step 10: Clean up test class
    r8 = agent_hub.execute_tool("z2yy_delete_class", {"class_name": renamed_class})
    assert_true(r8.get("success") is True, f"z2yy_delete_class removed {renamed_class}")


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("STARTING ZERO2YOLOYARD AGENT SUBSYSTEM TEST SUITE")
    print("#"*70)

    test_registry_integrity()
    test_agent_rest_endpoints()
    test_agent_cli_dispatcher()
    test_autonomous_workflow_lifecycle()

    print("\n" + "#"*70)
    print("ALL AGENT SUBSYSTEM AUDITS PASSED PERFECTLY (100% GREEN)")
    print("#"*70 + "\n")
