from graphviz import Digraph

# Create Digraph
dot = Digraph("ResearchFlowchart", format="png")
dot.attr(rankdir="TB", size="8")

# Start
dot.node("start", "Start", shape="ellipse", style="filled", fillcolor="lightgrey")

# Input Data (Direct Data Icon)
dot.node("data", "Bird Sound Spectrogram", shape="cylinder", style="filled", fillcolor="lightblue")

# ResNet & EfficientNet Processes
dot.node("resnet", "ResNet\n(Output: Probabilities Array)", shape="box", style="rounded,filled", fillcolor="lightyellow")
dot.node("effnet", "EfficientNet\n(Output: Probabilities Array)", shape="box", style="rounded,filled", fillcolor="lightyellow")

# Scalars
dot.node("resnet_scaled", "ResNet Probabilities × 0.6", shape="box", style="filled", fillcolor="white")
dot.node("effnet_scaled", "EffNet Probabilities × 0.4", shape="box", style="filled", fillcolor="white")

# Combine results
dot.node("combine", "Add Arrays", shape="diamond", style="filled", fillcolor="lightgreen")
dot.node("normalize", "Divide by 2 (Scalar)", shape="box", style="filled", fillcolor="white")
dot.node("argmax", "Bird Species with Greatest Probability", shape="ellipse", style="filled", fillcolor="lightpink")

# Edges
dot.edge("start", "data")
dot.edge("data", "resnet")
dot.edge("data", "effnet")
dot.edge("resnet", "resnet_scaled")
dot.edge("effnet", "effnet_scaled")
dot.edge("resnet_scaled", "combine")
dot.edge("effnet_scaled", "combine")
dot.edge("combine", "normalize")
dot.edge("normalize", "argmax")

# Render
file_path = "research_flowchart"
dot.render(file_path, cleanup=True)

file_path + ".png"
