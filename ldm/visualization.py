import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F


class LayerDataCollector():
    def __init__(self, name, layer_num, layer_type):
        self.name = name
        self.layer_num = layer_num
        self.layer_type = layer_type
        self.activations = []

    def __call__(self, module, input, output):
        # Collect the output activations
        self.activations.append(output.detach().cpu())

def register_data_collector_hooks(model, args=None):
    layer_hooks = []
    layer_num = 0
    layer_count = 0
    for name, m in model.named_modules():
        # Collect activations for linear and conv layers only
        if args.activations and isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            # print(str(layer_num) + "\n")
            # Initialize the data collector with the layer name, number, and type
            layer_type = type(m).__name__
            hook = LayerDataCollector(name, layer_num, layer_type)
            handle = m.register_forward_hook(hook)
            layer_hooks.append((hook, handle))
            layer_count += 1
        # elif args.plot_layer is None or \
        #    args.plot_layer == layer_num or \
        #    (isinstance(args.plot_layer, str) and args.plot_layer in name) or \
        #    (isinstance(args.plot_layer, list) and layer_num in args.plot_layer):
        #     m.plot = True
        #     m.plot_this_batch = True

        #     # Initialize the data collector with the layer name
        #     hook = LayerDataCollector(name)
        #     handle = m.register_forward_hook(hook)
        #     layer_hooks.append((hook, handle))
        layer_num += 1
    print (layer_count)
    return layer_hooks

# Collects activation stats and returns it in a pd dataframe
def collect_activation_statistics(layer_hooks):
    import pandas as pd

    # Collect data and layer info
    stats_list = []
    count = 0
    for (hook, handle) in layer_hooks:
        print(f"new hook {count} \n")
        # Remove the hook
        handle.remove()

        activations_list = hook.activations
        if len(activations_list) == 0:
            continue
        # Combine all collected activations for the layer (list of tensors)
        activations_tensor = torch.cat(activations_list, dim=0)
        activations_np = activations_tensor.numpy().flatten()

        # Calculate statistical metrics
        stats = {
            'layer_num': hook.layer_num,
            'layer_name': hook.name,
            'layer_type': hook.layer_type,
            'mean': np.mean(activations_np),
            'std': np.std(activations_np),
            'min': np.min(activations_np),
            'max': np.max(activations_np),
            'percentile_2': np.percentile(activations_np, 2),
            'percentile_5': np.percentile(activations_np, 5),
            'percentile_95': np.percentile(activations_np, 95),
            'percentile_98': np.percentile(activations_np, 98),
            'percentile_99.5': np.percentile(activations_np, 99.5),
        }
        stats_list.append(stats)
        count += 1

    # Create pandas DataFrame
    df = pd.DataFrame(stats_list)
    # Set index to layer number
    df.set_index('layer_num', inplace=True)
    return df


def plot_activations(model, layer_hooks, i=0, timestep=None, save_dir=None, args=None):
    import matplotlib.pyplot as plt
    import os
    import torch
    import numpy as np

    # Collect data and layer names
    data = []
    layer_names = []
    for (hook, handle) in layer_hooks:
        handle.remove()
        data.append(hook.layer_data)
        layer_names.append(hook.name)

    assert data != []

    # Number of layers to plot
    N = len(data)

    # Set up the figure with one subplot per layer
    fig, axs = plt.subplots(nrows=N, ncols=1, figsize=(10, N * 3))

    # Ensure axs is iterable
    if N == 1:
        axs = [axs]

    for idx, ax in enumerate(axs):
        activations_list = data[idx]
        layer_name = layer_names[idx]

        # Combine all collected activations for the layer (list of tensors)
        activations_tensor = torch.cat(activations_list, dim=0)

        # Convert activations to numpy array
        activations_np = activations_tensor.numpy().flatten()

        # Calculate statistical metrics
        mean_val = np.mean(activations_np)
        std_val = np.std(activations_np)
        min_val = np.min(activations_np)
        max_val = np.max(activations_np)

        # Calculate weights for frequency normalization
        weights = np.ones_like(activations_np) / len(activations_np)

        # Plot histogram of activations with normalized frequency
        bins = 256
        ax.hist(activations_np, bins=bins, weights=weights, log=args.log,
                color='lightgray', alpha=0.7)

        # Add vertical lines for min and max values
        ax.axvline(min_val, color='red', linestyle='dashed', linewidth=1, label=f"Min: {min_val:.2e}")
        ax.axvline(max_val, color='green', linestyle='dashed', linewidth=1, label=f"Max: {max_val:.2e}")

        # Add vertical lines for mean and standard deviations
        ax.axvline(mean_val, color='orange', linestyle='solid', linewidth=1.5, label=f"Mean: {mean_val:.2e}")
        ax.axvline(mean_val - std_val, color='purple', linestyle='dashed', linewidth=1, label=f"-1σ: {(mean_val - std_val):.2e}")
        ax.axvline(mean_val + std_val, color='purple', linestyle='dashed', linewidth=1, label=f"+1σ: {(mean_val + std_val):.2e}")
        ax.axvline(mean_val - 2*std_val, color='gray', linestyle='dashed', linewidth=1, label=f"-2σ: {(mean_val - 2*std_val):.2e}")
        ax.axvline(mean_val + 2*std_val, color='gray', linestyle='dashed', linewidth=1, label=f"+2σ: {(mean_val + 2*std_val):.2e}")

        # Set y-axis label to 'Frequency'
        ax.set_ylabel('Frequency')

        # Set title including layer name, batch number, and timestep
        ax.set_title(f"Layer: {layer_name}, Batch: {i}, Timestep: {timestep}")

        # Add legend to show statistical lines
        ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()

    # Save or show the plot
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = f"activations_batch_{i}_timestep_{timestep}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=args.plot_dpi if args and hasattr(args, 'plot_dpi') else 100, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
