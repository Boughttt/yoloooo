from ultralytics import YOLO

# 加载模型（第一次会自动下载）
model = YOLO("yolov8n.pt")

# 运行检测
results = model("test.png", save=True)

print("YOLO检测完成")