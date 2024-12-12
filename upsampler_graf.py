from graphviz import Digraph

# Define the corrected diagram with explicit interpolation step and sizes
def create_upsampler_with_interpolation():
    dot = Digraph("Upsampler_Architecture", format="png")

    # Step 1: Inputs
    dot.node(
        "Inputs",
        "Inputs:\n"
        "- Low-Resolution Image (3xH_lowxW_low)\n"
        "- Noise Embedding (1xCond)\n"
        "- Previous Frames (3xH_lowxW_low)",
        shape="box",
        style="filled",
        color="lightblue",
    )

    # Step 2: Interpolation
    dot.node(
        "Interpolation",
        "Bicubic Interpolation:\n(3xH_lowxW_low -> 3xH_highxW_high)",
        shape="box",
        style="filled",
        color="lightgray",
    )
    dot.edge("Inputs", "Interpolation", label="Upsampling Factor = 4")

    # Step 3: Denoiser
    dot.node(
        "Denoiser",
        "Denoiser:\n"
        "Conditioners + Noise Adjustments\n"
        "(Input: 3xH_highxW_high -> Output: 64xH_highxW_high)",
        shape="box",
        style="filled",
        color="lightgreen",
    )
    dot.edge("Interpolation", "Denoiser", label="Interpolated Frames")

    # Step 4: Downsampling Path
    dot.node(
        "Downsampling",
        "Downsampling Path:\n"
        "- ResBlock(2x Conv3x3, 64xH_highxW_high -> 64xH_high/2xW_high/2)\n"
        "- ResBlock(2x Conv3x3, 64xH_high/2xW_high/2 -> 128xH_high/4xW_high/4)",
        shape="box",
        style="filled",
        color="khaki",
    )
    dot.edge("Denoiser", "Downsampling", label="Rescaled Noise + Conditioners")

    # Step 5: Bottleneck
    dot.node(
        "Bottleneck",
        "Bottleneck:\n"
        "- ResBlock(2x Conv3x3 + Self-Attention, 128xH_high/4xW_high/4 -> 256xH_high/8xW_high/8)",
        shape="box",
        style="filled",
        color="orange",
    )
    dot.edge("Downsampling", "Bottleneck")

    # Step 6: Upsampling Path
    dot.node(
        "Upsampling",
        "Upsampling Path:\n"
        "- ResBlock(2x Conv3x3, 256xH_high/8xW_high/8 -> 128xH_high/4xW_high/4)\n"
        "- ResBlock(2x Conv3x3, 128xH_high/4xW_high/4 -> 64xH_high/2xW_high/2)",
        shape="box",
        style="filled",
        color="lightpink",
    )
    dot.edge("Bottleneck", "Upsampling")

    # Step 7: Output Processing
    dot.node(
        "OutputProcessing",
        "Output Processing:\nClamp to [-1, 1]\n(Output: 64xH_highxW_high -> 3xH_highxW_high)",
        shape="box",
        style="filled",
        color="lightyellow",
    )
    dot.edge("Upsampling", "OutputProcessing")

    # Step 8: Output
    dot.node(
        "Output",
        "Output:\nHigh-Resolution Image (3xH_highxW_high)",
        shape="box",
        style="filled",
        color="lightblue",
    )
    dot.edge("OutputProcessing", "Output")

    # Render and return diagram
    dot.render("Upsampler_Architecture_Interpolation", format="png", cleanup=True)

create_upsampler_with_interpolation()
