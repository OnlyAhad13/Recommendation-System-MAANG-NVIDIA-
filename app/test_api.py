#!/usr/bin/env python3
"""
Simple test script for the recommendation API.

Usage:
    python app/test_api.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n🔍 Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ Health check passed")

def test_root():
    """Test root endpoint."""
    print("\n🔍 Testing / endpoint...")
    response = requests.get(BASE_URL)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ Root endpoint passed")

def test_recommendations():
    """Test recommendation endpoint."""
    print("\n🔍 Testing /recommend endpoint...")
    
    payload = {
        "user_id": "0",  # Use first user in vocab
        "k": 5,
        "exclude_seen": True
    }
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/recommend",
        json=payload
    )
    latency = (time.time() - start_time) * 1000
    
    print(f"Status: {response.status_code}")
    print(f"Latency: {latency:.2f}ms")
    
    if response.status_code == 200:
        data = response.json()
        print(f"User: {data['user_id']}")
        print(f"Recommendations: {data['count']}")
        print("\nTop 5 items:")
        for rec in data['recommendations'][:5]:
            print(f"  Rank {rec['rank']}: Item {rec['item_id']} (score: {rec['score']:.4f})")
        print("✅ Recommendations test passed")
    else:
        print(f"❌ Error: {response.text}")

def test_scoring():
    """Test scoring endpoint."""
    print("\n🔍 Testing /score endpoint...")
    
    payload = {
        "user_id": "0",
        "item_ids": ["0", "1", "2"]
    }
    
    response = requests.post(
        f"{BASE_URL}/score",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"User: {data['user_id']}")
        print("Scores:")
        for item_id, score in data['scores'].items():
            print(f"  Item {item_id}: {score:.4f}")
        print("✅ Scoring test passed")
    else:
        print(f"❌ Error: {response.text}")

def test_model_info():
    """Test model info endpoint."""
    print("\n🔍 Testing /model/info endpoint...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Model Info:")
        print(f"  Version: {data['version']}")
        print(f"  Users: {data['num_users']}")
        print(f"  Items: {data['num_items']}")
        print(f"  Embedding dim: {data['embedding_dim']}")
        print("✅ Model info test passed")
    else:
        print(f"❌ Error: {response.text}")

def test_batch_recommendations():
    """Test batch recommendations."""
    print("\n🔍 Testing /recommend/batch endpoint...")
    
    params = {
        "user_ids": ["0", "1"],
        "k": 3
    }
    
    response = requests.post(
        f"{BASE_URL}/recommend/batch",
        params=params
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Processed {data['count']} users")
        for user_data in data['recommendations']:
            print(f"  User {user_data['user_id']}: {len(user_data['recommendations'])} recs")
        print("✅ Batch recommendations test passed")
    else:
        print(f"❌ Error: {response.text}")

def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("🧪 TESTING RECOMMENDATION API")
    print("="*60)
    
    try:
        test_health()
        test_root()
        test_model_info()
        test_recommendations()
        test_scoring()
        test_batch_recommendations()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the API is running:")
        print("  uvicorn app.main:app --host 0.0.0.0 --port 8000")
    
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
