from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_file = "sd3_xray_training/events.out.tfevents.1764639164.aiaq82ekvr2po-0.44314.0"   # 你的文件名
output_file = "log.txt"

ea = EventAccumulator(event_file)
ea.Reload()

tags = ea.Tags().get("scalars", [])

with open(output_file, "w", encoding="utf-8") as f:
    f.write("Scalars found:\n")
    for tag in tags:
        f.write(f"\n[{tag}]\n")
        events = ea.Scalars(tag)
        for e in events:
            f.write(f"step={e.step}, value={e.value}\n")

print("已写入 log.txt")
