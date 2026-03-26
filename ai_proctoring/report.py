import matplotlib.pyplot as plt
from collections import Counter
import os

def generate_report(violations):
    if not violations:
        print("\n" + "="*40)
        print("SESSION SUMMARY")
        print("="*40)
        print("Total Violations: 0")
        print("No violations detected. Great job!")
        print("="*40)
        return

    print("\n" + "="*40)
    print("SESSION SUMMARY")
    print("="*40)
    
    total_violations = len(violations)
    print(f"Total Violations: {total_violations}")
    
    types = [v['violation'] for v in violations]
    type_counts = Counter(types)
    
    print("\nViolation Breakdown:")
    for v_type, count in type_counts.items():
        print(f"- {v_type}: {count}")
        
    print("="*40)
    
    # Generate Graph
    timestamps = [v['timestamp'] for v in violations]
    
    plt.figure(figsize=(10, 6))
    
    # Group by timestamp (second)
    time_counts = Counter(timestamps)
    
    times = sorted(list(time_counts.keys()))
    counts = [time_counts[t] for t in times]
    
    plt.bar(times, counts, color='red', width=1.0)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Violations')
    plt.title('Proctoring Session Violations Over Time')
    plt.grid(axis='y', alpha=0.75)
    
    # Save graph
    report_path = 'violations_report.png'
    plt.savefig(report_path)
    print(f"Graph saved as '{report_path}'")
    
    try:
        # Prevent blocking if run headless, but show if possible
        plt.show(block=False)
        plt.pause(3)
    except Exception:
        pass
    finally:
        plt.close()
