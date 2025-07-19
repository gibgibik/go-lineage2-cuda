import onnx

model = onnx.load("dbnet.onnx")

print("🔵 Inputs:")
for i in model.graph.input:
    print(" -", i.name)

print("🟢 Outputs:")
for o in model.graph.output:
    print(" -", o.name)