"""End-to-end test for kauto agent system."""

import json
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_mcp_command(method: str, params: dict = None) -> dict:
    """Run MCP server command."""
    cmd = ["python", "core/agent/mcp_server.py", "--method", method]
    if params:
        cmd.extend(["--params", json.dumps(params)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
    return json.loads(result.stdout)


def test_mcp_server():
    """Test MCP server functionality."""
    print("=" * 80)
    print("Testing MCP Server")
    print("=" * 80)
    
    # Test 1: List tools
    print("\n1. Testing list_tools...")
    result = run_mcp_command("list_tools")
    assert result["success"], "list_tools failed"
    assert len(result["tools"]) == 8, f"Expected 8 tools, got {len(result['tools'])}"
    print("✅ list_tools passed")
    
    # Test 2: Set competition
    print("\n2. Testing set_competition...")
    result = run_mcp_command("set_competition", {"competition_name": "titanic"})
    assert result["success"], "set_competition failed"
    assert result["competition"] == "titanic"
    print("✅ set_competition passed")
    
    # Test 3: Read file
    print("\n3. Testing read_file...")
    result = run_mcp_command("read_file", {"file_path": "docs/README.md", "competition_name": "titanic"})
    assert result["success"], f"read_file failed: {result.get('error', 'Unknown error')}"
    assert "content" in result
    assert len(result["content"]) > 0
    print("✅ read_file passed")
    
    # Test 4: Get experiment history
    print("\n4. Testing get_experiment_history...")
    result = run_mcp_command("get_experiment_history", {"n": 5, "competition_name": "titanic"})
    assert result["success"], f"get_experiment_history failed: {result.get('error', 'Unknown error')}"
    assert "experiments" in result
    print(f"✅ get_experiment_history passed (found {len(result['experiments'])} experiments)")
    
    # Test 5: Get best experiment
    print("\n5. Testing get_best_experiment...")
    result = run_mcp_command("get_best_experiment", {"metric_name": "cv_score", "competition_name": "titanic"})
    # This might fail if no experiments exist, which is okay
    if result["success"]:
        print(f"✅ get_best_experiment passed (best score: {result['best_experiment']['metrics']['cv_score']})")
    else:
        print("⚠️  get_best_experiment: No experiments found (expected for fresh setup)")
    
    print("\n" + "=" * 80)
    print("MCP Server Tests Completed")
    print("=" * 80)


def test_experiment_runner():
    """Test experiment runner."""
    print("\n" + "=" * 80)
    print("Testing Experiment Runner")
    print("=" * 80)
    
    from core.experiment.runner import ExperimentRunner
    
    runner = ExperimentRunner(Path.cwd())
    
    print("\n1. Testing experiment execution...")
    result = runner.run(
        competition_name="titanic",
        exp_script="experiments/exp001_baseline.py"
    )
    
    assert result["success"], f"Experiment failed: {result.get('error', 'Unknown error')}"
    assert "duration_seconds" in result
    print(f"✅ Experiment executed successfully in {result['duration_seconds']:.2f}s")
    
    print("\n" + "=" * 80)
    print("Experiment Runner Tests Completed")
    print("=" * 80)


def test_experiment_tracker():
    """Test experiment tracker."""
    print("\n" + "=" * 80)
    print("Testing Experiment Tracker")
    print("=" * 80)
    
    from core.experiment.tracker import ExperimentTracker
    
    comp_dir = Path.cwd() / "competitions" / "titanic"
    tracker = ExperimentTracker(comp_dir)
    
    print("\n1. Testing experiment logging...")
    tracker.log_experiment(
        exp_name="test_exp",
        config={"test": True},
        metrics={"cv_score": 0.85},
        success=True,
        notes="Test experiment"
    )
    print("✅ Experiment logged successfully")
    
    print("\n2. Testing get_recent_experiments...")
    recent = tracker.get_recent_experiments(n=3)
    assert len(recent) > 0, "No experiments found"
    print(f"✅ Found {len(recent)} recent experiments")
    
    print("\n3. Testing get_best_experiment...")
    best = tracker.get_best_experiment()
    if best:
        print(f"✅ Best experiment: {best['exp_name']} (score: {best['metrics']['cv_score']})")
    else:
        print("⚠️  No successful experiments found")
    
    print("\n" + "=" * 80)
    print("Experiment Tracker Tests Completed")
    print("=" * 80)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("KAUTO END-TO-END TESTS")
    print("=" * 80)
    
    try:
        # Test 1: MCP Server
        test_mcp_server()
        
        # Test 2: Experiment Runner
        test_experiment_runner()
        
        # Test 3: Experiment Tracker
        test_experiment_tracker()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
