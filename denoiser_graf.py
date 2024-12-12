import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def add_block(ax, x, y, width, height, label, color="lightblue"):
    """Add a block to the diagram."""
    rect = Rectangle((x, y), width, height, edgecolor="black", facecolor=color, lw=1.5)
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        fontsize=8,
        ha="center",
        va="center",
        wrap=True,
    )

def draw_full_model_hierarchical_diagram():
    fig, ax = plt.subplots(figsize=(18, 14))

    # Outer Block (Full Model)
    add_block(ax, -1, 8, 20, 5, "Full Model: Predict Next Frame\n(Integrating Denoiser, Sampler, Inner Model)", color="lightgray")

    # Step 1: Inputs
    add_block(ax, 0, 9.5, 3, 1, "Inputs:\nPast Frames\n(Nx3xHxW)", color="lightblue")
    add_block(ax, 0, 8.5, 3, 1, "Inputs:\nPast Actions\n(NxNumActions)", color="lightblue")
    add_block(ax, 0, 7.5, 3, 1, "Inputs:\nNoise Embedding\n(1xCond)", color="lightblue")

    # Step 2: Denoiser
    add_block(ax, 4, 8, 6, 3, "Denoiser:\nConditioners, Noise Adjustments\n(Add Sigma Noise)", color="lightgreen")

    # Step 3: Diffusion Sampler
    add_block(ax, 11, 8, 6, 3, "Diffusion Sampler:\nStepwise Prediction\n(Sigma-Controlled Sampling)", color="khaki")

    # Step 4: Inner Model
    add_block(ax, 11, 4.5, 6, 2, "Inner Model:\nUNet + Residual Blocks\n(Image Transformations)", color="orange")

    # Step 5: Outputs
    add_block(ax, 18, 7.5, 3, 1, "Predicted Frame\n(3xHxW)", color="lightblue")

    # Wiring within the diagram
    ax.annotate("", xy=(3, 9), xytext=(4, 9), arrowprops=dict(arrowstyle="->"))  # Inputs to Denoiser
    ax.annotate("", xy=(10, 9), xytext=(11, 9), arrowprops=dict(arrowstyle="->"))  # Denoiser to Sampler
    ax.annotate("", xy=(14, 7), xytext=(14, 6.5), arrowprops=dict(arrowstyle="->"))  # Sampler to Inner Model
    ax.annotate("", xy=(17, 6), xytext=(18, 8), arrowprops=dict(arrowstyle="->"))  # Inner Model to Predicted Frame

    # Nesting components in Denoiser
    add_block(ax, 5, 9.5, 2, 0.5, "Conditioners\n(c_in, c_skip, c_out)", color="lightyellow")
    add_block(ax, 5, 8.5, 2, 0.5, "Noise Adjustments\n(Add Sigma Noise)", color="lightpink")

    # Formatting
    ax.set_xlim(-2, 22)
    ax.set_ylim(4, 13)
    ax.axis("off")
    plt.title("Full Model Hierarchical Block Diagram\n(Predicting Next Frame from Past Actions and Frames)", fontsize=16)
    plt.show()

# Call the function to draw the hierarchical full model diagram
draw_full_model_hierarchical_diagram()
