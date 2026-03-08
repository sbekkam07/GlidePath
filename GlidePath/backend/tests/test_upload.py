import requests
import os

# Configuration
BASE_URL = "http://127.0.0.1:8000"
TEST_VIDEO_PATH = "test_landing.mp4"

def test_analyze_video():
    # 1. Ensure a dummy test video exists
    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"Creating a dummy test file: {TEST_VIDEO_PATH}...")
        with open(TEST_VIDEO_PATH, "wb") as f:
            f.write(b"dummy video content for testing")

    # 2. Prepare the file for upload
    print(f"Uploading {TEST_VIDEO_PATH} to {BASE_URL}/analyze-video...")
    
    try:
        with open(TEST_VIDEO_PATH, "rb") as f:
            files = {"file": (TEST_VIDEO_PATH, f, "video/mp4")}
            response = requests.post(f"{BASE_URL}/analyze-video", files=files)
        
        # 3. Check results
        if response.status_code == 200:
            print("\n✅ Success!")
            print("Response Data:")
            print(response.json())
        else:
            print(f"\n❌ Failed with status code: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the server. Make sure uvicorn is running!")

if __name__ == "__main__":
    test_analyze_video()
