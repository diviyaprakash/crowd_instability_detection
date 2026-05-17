import time
from app import run_pipeline, processing_state

def test():
    print("Starting pipeline...")
    # call it synchronously
    run_pipeline("Delhi Metro Crowd #shorts.mp4", 0.5, 0.35)
    print("Pipeline finished.")
    print("Events generated:", len(processing_state["events"]))
    for e in processing_state["events"][:5]:
        print(e)

if __name__ == "__main__":
    test()
