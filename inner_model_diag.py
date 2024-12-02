import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_inner_model_architecture_diagram():
    # Create a new figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Add rectangles for each layer
    ax.add_patch(patches.Rectangle((0.1, 0.8), 0.3, 0.1, edgecolor='black', facecolor='lightblue'))
    ax.text(0.25, 0.85, 'Input Tensor', ha='center', va='center')

    ax.add_patch(patches.Rectangle((0.1, 0.65), 0.3, 0.1, edgecolor='black', facecolor='lightgreen'))
    ax.text(0.25, 0.7, 'Conv Layer 1\n(64 filters, 3x3)', ha='center', va='center')

    ax.add_patch(patches.Rectangle((0.1, 0.5), 0.3, 0.1, edgecolor='black', facecolor='lightgreen'))
    ax.text(0.25, 0.55, 'ReLU Activation', ha='center', va='center')

    ax.add_patch(patches.Rectangle((0.1, 0.35), 0.3, 0.1, edgecolor='black', facecolor='lightgreen'))
    ax.text(0.25, 0.4, 'Conv Layer 2\n(64 filters, 3x3)', ha='center', va='center')

    ax.add_patch(patches.Rectangle((0.1, 0.2), 0.3, 0.1, edgecolor='black', facecolor='lightgreen'))
    ax.text(0.25, 0.25, 'ReLU Activation', ha='center', va='center')

    ax.add_patch(patches.Rectangle((0.1, 0.05), 0.3, 0.1, edgecolor='black', facecolor='lightgreen'))
    ax.text(0.25, 0.1, 'Conv Layer 3\n(128 filters, 3x3)', ha='center', va='center')

    ax.add_patch(patches.Rectangle((0.5, 0.35), 0.3, 0.1, edgecolor='black', facecolor='lightyellow'))
    ax.text(0.65, 0.4, 'Skip Connection', ha='center', va='center')

    ax.add_patch(patches.Rectangle((0.5, 0.2), 0.3, 0.1, edgecolor='black', facecolor='lightgreen'))
    ax.text(0.65, 0.25, 'Conv Layer 4\n(256 filters, 3x3)', ha='center', va='center')

    ax.add_patch(patches.Rectangle((0.5, 0.05), 0.3, 0.1, edgecolor='black', facecolor='lightcoral'))
    ax.text(0.65, 0.1, 'Output Tensor', ha='center', va='center')

    # Add arrows to indicate flow
    ax.annotate('', xy=(0.25, 0.8), xytext=(0.25, 0.75),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.25, 0.65), xytext=(0.25, 0.6),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.25, 0.5), xytext=(0.25, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.25, 0.35), xytext=(0.25, 0.3),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.25, 0.2), xytext=(0.25, 0.15),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.25, 0.05), xytext=(0.25, 0),
                arrowprops=dict(arrowstyle='->', lw=2))

    # Add arrows for skip connection
    ax.annotate('', xy=(0.25, 0.35), xytext=(0.65, 0.35),
                arrowprops=dict(arrowstyle='->', lw=2))

    # Set limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1)
    ax.axis('off')

    # Title
    plt.title('Denoiser Inner Model Architecture Diagram', fontsize=16)

    # Show the diagram
    plt.show()

# Call the function to draw the diagram
draw_inner_model_architecture_diagram()