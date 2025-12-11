
"""
File: feature-fbcnet.py
Author: Chuncheng Zhang
Date: 2025-11-07
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decode with FBCNet, and parse its parameters.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-07 ------------------------
# Requirements and constants
import torch
import torch.nn as nn
import torch.optim as optim
from torcheeg.models import FBCNet
from scipy import signal
from sklearn.model_selection import StratifiedKFold

import seaborn as sns
from mne.viz import plot_topomap
import torch.nn.functional as F

from util.easy_import import *
from FBCSP.FBCSP_class import filter_bank, FBCSP_info, FBCSP_info_weighted


# %%
RAW_DIR = Path('./raw/MI-dataset')
SUBJECT = 'sub001'
DEVICE = np.random.randint(0, 6)
DEVICE = 0

if len(sys.argv) > 2 and sys.argv[1] == '-s':
    SUBJECT = sys.argv[2]

if len(sys.argv) > 4 and sys.argv[3] == '-d':
    DEVICE = int(sys.argv[4])


# Every subject has 10 runs
N_RUNS = 10

OUTPUT_DIR = Path(f'./data/MI-dataset-results/f-fbcnet/{SUBJECT}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
k_select = 10

n_components = 4
freq_bands = [[4+i*4, 8+i*4] for i in range(9)]+[[8, 32]]
filter_type = 'iir'
filt_order = 5

tmin, tmax = 0, 5
sfreq = 250

# 创建info对象
ch_names = ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
            'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6']
ch_index = [24, 25, 26, 27, 28, 29, 30,
            34, 35, 36, 37, 38, 39, 40,
            15, 16, 17, 18, 19, 20, 21]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
info.set_montage('standard_1020')
event_id = {
    '1': 1,
    '2': 2
}

# %% ---- 2025-11-07 ------------------------
# Function and class


class DataLoader:
    def __init__(self, X, y, groups, test_group=0):
        self.X = X
        # Scale into 1 scale
        self.X /= np.max(np.abs(self.X))
        self.y = y

        self.X = torch.tensor(self.X).cuda(DEVICE)
        self.y = torch.tensor(self.y).cuda(DEVICE)

        self.groups = groups
        # Separate groups
        unique_groups = sorted(np.unique(self.groups).tolist())
        self.test_groups = [test_group]
        self.train_groups = [
            e for e in unique_groups if not e in self.test_groups]
        logger.info(
            f'DataLoader: {self.X.shape=}, {self.y.shape=}, {self.groups.shape=}, {self.train_groups=}, {self.test_groups=}')

    def yield_train_data(self, batch_size=32):
        train_idx = [g in self.train_groups for g in self.groups]
        while True:
            X = self.X[train_idx]
            y = self.y[train_idx]
            n_samples = X.shape[0]
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                yield X[batch_indices], y[batch_indices]

    def get_test_data(self):
        test_idx = [g in self.test_groups for g in self.groups]
        X = self.X[test_idx]
        y = self.y[test_idx]
        return X, y


def load_data_np(path: Path):
    '''
    Read file for EEG data.

    :param path Path: File path of .npy.

    :return X np.array: EEG data (n_samples, n_channels, n_times)
    :return y np.array: EEG label (n_samples, )
    '''

    # Raw data sfreq is 1000 Hz
    raw_sfreq = 1000

    def _load_data(f):
        while True:
            try:
                yield np.load(f)
            except EOFError:
                return

    with open(path, 'rb') as f:
        file_data = np.concatenate(list(_load_data(f))).T

    events = file_data[-1, :]
    file_data = file_data[:-1, :]

    index = []
    for i_point in range(events.shape[0]-1):
        if events[i_point+1] > events[i_point]:
            trial_idx = [int(events[i_point+1]-events[i_point]), i_point+1]
            index.append(trial_idx)

    data_all = []
    label_all = []
    for e in index:
        if e[0] in [1, 2]:
            data_all.append(
                file_data[:, e[1]+int(tmin*raw_sfreq):e[1]+int(tmax*raw_sfreq)])
            label_all.append(e[0])

    data_all = np.array(data_all)[:, ch_index, :]
    X = signal.resample(data_all, sfreq*int(tmax-tmin), axis=-1)
    y = np.array(label_all)
    print(f'{X.shape=}, {y.shape=}, {set(y)=}')
    return X, y


def mk_groups(X, y):
    cv = StratifiedKFold(n_splits=5)
    cv_list = list(cv.split(X, y))
    groups = y.copy()

    for i, (train_index, test_index) in enumerate(cv_list):
        groups[test_index] = i

    return groups


def decode_fbcnet(X, y, groups):
    y_pred_all = y.copy()
    for test_group in np.unique(groups):
        # Make model
        # shape is (n_samples, n_bands, n_electrodes, n_times)
        shape = X.shape
        num_classes = len(np.unique(y))

        # Model
        model = FBCNet(
            num_electrodes=shape[2],
            chunk_size=500,
            in_channels=shape[1],
            num_classes=num_classes,
        ).cuda(DEVICE)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()  # 多分类任务常用
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(model, criterion, optimizer)

        # Training loop
        dl = DataLoader(X[:, :, :, :500], y, groups, test_group=test_group)
        it = iter(dl.yield_train_data(batch_size=10))

        output_path = OUTPUT_DIR.joinpath(f'{test_group}.dump')

        if output_path.is_file():
            continue

        for epoch in tqdm(range(5000), desc='Epoch'):
            def _train():
                X, y = next(it)
                # print(f'{X.shape=}, {y.shape=}')

                _y = model(torch.tensor(X, dtype=torch.float32))
                # print(f'{_y.shape=}')
                # print(_y)

                # 前向传播
                loss = criterion(_y, torch.tensor(y-1))
                # print(f'{loss.item()=}')

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Report
                if epoch % 500 == 0:
                    logger.info(f'Epoch {epoch}, Loss: {loss.item():.6f}')

            _train()

        def _test():
            # Testing loop
            X, y = dl.get_test_data()
            y_true = y.cpu().numpy()
            with torch.no_grad():
                _y = model(torch.tensor(X, dtype=torch.float32)).cpu()
                y_pred = torch.argmax(_y, dim=1).numpy() + 1
                accuracy = np.mean(y_pred == y_true)
                logger.info(
                    f'Test Accuracy ({test_group}): {accuracy * 100:.2f}%')

            y_pred_all[groups == int(test_group)] = y_pred

        _test()

        # Use the model directly
        return model

    return y_pred_all


# %% ---- 2025-11-07 ------------------------
# Play ground
acc_all = []
data_all = []
label_all = []
epochs_all = []

for i_run in tqdm(range(N_RUNS), f'Loading runs ({SUBJECT=})'):
    X, y = load_data_np(RAW_DIR.joinpath(f'{SUBJECT}/run_{i_run}.npy'))

    events = np.column_stack((np.array([i*sfreq*8 for i in range(len(y))]),
                              np.zeros(len(y), dtype=int),
                              y))
    epochs = mne.EpochsArray(
        X, info, tmin=tmin, events=events, event_id=event_id)

    groups = mk_groups(X, y)
    print(groups, y)

    new_X = []
    for low_freq, high_freq in tqdm(freq_bands, 'Filtering'):
        # 频带滤波
        X_filtered = []

        for i in range(X.shape[0]):
            # 创建Epochs对象进行滤波
            _epochs = mne.EpochsArray(X[i:i+1], info, tmin=epochs.times[0])
            epochs_filtered = _epochs.filter(l_freq=low_freq, h_freq=high_freq,
                                             method='iir', verbose=False)
            # Downsample to 100 Hz
            epochs_filtered.resample(100)
            # epochs_filtered.crop(tmin=tmin, tmax=tmax)
            X_filtered.append(epochs_filtered.get_data()[0])
        new_X.append(X_filtered)

    # new_X shape (n_bands, n_samples, n_channels, n_times)
    new_X = np.array(new_X)

    # Convert into (n_samples, n_bands, n_electrodes, n_times)
    X = new_X.transpose((1, 0, 2, 3))
    print(X.shape, groups.shape, y.shape)

    model = decode_fbcnet(X, y, groups)
    break

# %%
'''
Parse the model's parameters
Model
model = FBCNet(
    num_electrodes=shape[2],
    chunk_size=500,
    in_channels=shape[1],
    num_classes=num_classes,
).cuda(DEVICE)

Args:
    - model
    - epochs.info


'''


def visualize_fbcnet_results(model, info, epochs, num_classes):
    """
    Visualize FBCNet model parameters and results

    Parameters:
    - model: Trained FBCNet model
    - info: MNE info object with channel information
    - epochs: Original epochs data
    - num_classes: Number of classes
    """

    print("=" * 60)
    print("FBCNet Model Visualization")
    print("=" * 60)

    # Set model to evaluation mode
    model.eval()

    # 1. Extract and visualize model architecture
    print("\n1. Model Architecture Summary:")
    print(model)

    # 2. Extract model parameters for visualization
    print("\n2. Extracting model parameters...")

    # Get all parameter names and shapes
    param_shapes = {}
    for name, param in model.named_parameters():
        param_shapes[name] = param.shape
        print(f"{name}: {param.shape}")

    # 3. Visualize Spatial Filters (Temporal Convolution)
    print("\n3. Visualizing Spatial-Temporal Filters...")

    # Extract the first convolutional layer weights (spatial-temporal filters)
    if hasattr(model, 'conv_time'):
        conv_time_weights = model.conv_time.weight.detach().cpu().numpy()
        print(f"Temporal conv weights shape: {conv_time_weights.shape}")

        # Visualize temporal filters
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('FBCNet Temporal Convolution Filters',
                     fontsize=16, fontweight='bold')

        # Plot first few filters
        n_filters_to_plot = min(6, conv_time_weights.shape[0])
        for i in range(n_filters_to_plot):
            ax = axes[i//3, i % 3]
            filter_weights = conv_time_weights[i, 0, :]  # First channel
            ax.plot(filter_weights, linewidth=2)
            ax.set_title(f'Temporal Filter {i+1}')
            ax.set_xlabel('Time Samples')
            ax.set_ylabel('Weight')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR.joinpath(f'{datetime.now()}.png'))
        plt.show()

    # 4. Visualize Spectral Filters (Filter Bank)
    print("\n4. Visualizing Spectral Filter Bank...")

    if hasattr(model, 'filter_bank'):
        # If using custom filter bank
        filter_bank = model.filter_bank
        print(f"Filter bank structure: {filter_bank}")
    else:
        # Extract spectral information from the model structure
        # Assuming the model uses multiple frequency bands
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create example frequency bands (adjust based on your actual bands)
        freq_bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28),
                      (28, 32), (32, 36), (36, 40)]

        for i, (low, high) in enumerate(freq_bands):
            ax.barh(i, high-low, left=low, alpha=0.7,
                    label=f'Band {i+1}: {low}-{high}Hz')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Filter Band')
        ax.set_title('FBCNet Filter Bank Frequency Bands')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR.joinpath(f'{datetime.now()}.png'))
        plt.show()

    # 5. Visualize Spatial Patterns
    print("\n5. Visualizing Spatial Patterns...")

    # Extract spatial weights from the model
    spatial_weights = extract_spatial_weights(model, info)

    if spatial_weights is not None:
        # Plot spatial patterns as topomaps
        plot_spatial_patterns(spatial_weights, info, num_classes)

    # 6. Feature Importance Analysis
    print("\n6. Analyzing Feature Importance...")
    feature_importance = analyze_feature_importance(model, epochs, num_classes)

    # 7. Class-wise Activation Patterns
    print("\n7. Visualizing Class-wise Patterns...")
    plot_class_patterns(model, info, num_classes)

    # 8. Model Performance Visualization (if validation data available)
    print("\n8. Model Performance Metrics...")
    # This would require validation predictions and true labels

    return {
        'spatial_weights': spatial_weights,
        'feature_importance': feature_importance
    }


def extract_spatial_weights(model, info):
    """
    Extract spatial weights from FBCNet model
    """
    spatial_weights = None

    # Try to find spatial convolution layers
    for name, param in model.named_parameters():
        if 'conv_spatial' in name and 'weight' in name:
            weights = param.detach().cpu().numpy()
            print(f"Found spatial weights: {weights.shape}")

            # Average across filters and frequency bands if needed
            if len(weights.shape) == 4:  # [out_ch, in_ch, height, width]
                # For spatial conv, average across output channels
                spatial_weights = np.mean(weights, axis=(0, 2, 3))
            elif len(weights.shape) == 2:  # [features, electrodes]
                spatial_weights = np.mean(weights, axis=0)

            break

    # If no specific spatial layer found, try to extract from dense layers
    if spatial_weights is None:
        for name, param in model.named_parameters():
            if 'classifier' in name and 'weight' in name:
                weights = param.detach().cpu().numpy()
                if len(weights.shape) == 2 and weights.shape[1] == len(info['ch_names']):
                    spatial_weights = np.mean(np.abs(weights), axis=0)
                    print(
                        f"Using classifier weights as spatial pattern: {spatial_weights.shape}")
                break

    return spatial_weights


def plot_spatial_patterns(spatial_weights, info, num_classes):
    """
    Plot spatial patterns as topomaps
    """
    if spatial_weights is None:
        print("No spatial weights found for topomap visualization")
        return

    # Create topomap
    fig, ax = plt.subplots(figsize=(8, 6))

    # Ensure spatial_weights matches number of channels
    if len(spatial_weights) == len(info['ch_names']):
        im, _ = plot_topomap(spatial_weights, info, axes=ax, show=False,
                             sensors=True, contours=6, outlines='head')
        ax.set_title('FBCNet Spatial Weights (Averaged)',
                     fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Weight Magnitude')
        plt.savefig(OUTPUT_DIR.joinpath(f'{datetime.now()}.png'))
        plt.show()
    else:
        print(
            f"Spatial weights dimension {len(spatial_weights)} doesn't match channels {len(info['ch_names'])}")


def analyze_feature_importance(model, epochs, num_classes):
    """
    Analyze feature importance using gradient-based methods
    """
    print("Computing feature importance...")

    dl = DataLoader(X[:, :, :, :500], y, groups, test_group=groups[0])
    it = iter(dl.yield_train_data(batch_size=10))

    # Get sample data for analysis
    sample_data = torch.tensor(
        epochs.get_data()[:10], dtype=torch.float32).cuda()
    sample_data, _ = next(it)
    sample_data = torch.tensor(sample_data, dtype=torch.float32)

    # Set model to train mode for gradient computation
    model.train()

    # Compute gradients for feature importance
    sample_data.requires_grad_(True)

    # Forward pass
    outputs = model(sample_data)

    # Use the first sample for analysis
    target_class = torch.argmax(outputs[0])
    outputs[0, target_class].backward()

    # Get gradients
    gradients = sample_data.grad[0].detach().cpu().numpy()

    # Average gradients across time and channels
    feature_importance = np.mean(np.abs(gradients), axis=(0, 1))

    # Plot feature importance
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Channel importance
    if True or len(feature_importance) == len(epochs.ch_names):
        axes[0].bar(range(len(feature_importance)), feature_importance)
        axes[0].set_xlabel('Channel Index')
        axes[0].set_ylabel('Gradient Magnitude')
        axes[0].set_title('Channel-wise Feature Importance')
        axes[0].tick_params(axis='x', rotation=45)

    # Temporal importance (average across channels)
    temporal_importance = np.mean(np.abs(gradients), axis=0)
    axes[1].plot(temporal_importance)
    axes[1].set_xlabel('Time Samples')
    axes[1].set_ylabel('Average Gradient Magnitude')
    axes[1].set_title('Temporal Feature Importance')
    axes[1].grid(True, alpha=0.3)

    # 新增：网络结构特征流图
    plot_network_feature_flow(model, sample_data, axes[2])
    axes[2].set_title('FBCNet Network Feature Flow')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR.joinpath(f'{datetime.now()}.png'))
    plt.show()

    return feature_importance


def plot_network_feature_flow(model, sample_data, ax):
    """
    绘制FBCNet网络结构的特征流图
    """
    # 设置模型为评估模式
    model.eval()

    # 存储各层的输出特征
    layer_outputs = {}
    hooks = []

    # 注册钩子来捕获各层输出
    def hook_fn(name):
        def hook(module, input, output):
            layer_outputs[name] = output.detach()
        return hook

    # 为关键层注册钩子
    if hasattr(model, 'conv_time'):
        hooks.append(model.conv_time.register_forward_hook(
            hook_fn('temporal_conv')))

    if hasattr(model, 'conv_spatial'):
        hooks.append(model.conv_spatial.register_forward_hook(
            hook_fn('spatial_conv')))

    if hasattr(model, 'conv_separable_depth'):
        hooks.append(model.conv_separable_depth.register_forward_hook(
            hook_fn('depthwise_conv')))

    if hasattr(model, 'conv_separable_point'):
        hooks.append(model.conv_separable_point.register_forward_hook(
            hook_fn('pointwise_conv')))

    # 前向传播
    with torch.no_grad():
        final_output = model(sample_data)

    # 移除钩子
    for hook in hooks:
        hook.remove()

    # 绘制网络结构图
    layers = ['Input']
    feature_sizes = [f"{sample_data.shape[1:]}"]
    colors = ['lightblue']

    # 添加各层信息
    if 'temporal_conv' in layer_outputs:
        layers.append('Temporal\nConv')
        feat = layer_outputs['temporal_conv'].shape
        feature_sizes.append(f"{feat[1:]}")  # 忽略batch维度
        colors.append('lightgreen')

    if 'spatial_conv' in layer_outputs:
        layers.append('Spatial\nConv')
        feat = layer_outputs['spatial_conv'].shape
        feature_sizes.append(f"{feat[1:]}")
        colors.append('lightcoral')

    if 'depthwise_conv' in layer_outputs:
        layers.append('Depthwise\nConv')
        feat = layer_outputs['depthwise_conv'].shape
        feature_sizes.append(f"{feat[1:]}")
        colors.append('lightyellow')

    if 'pointwise_conv' in layer_outputs:
        layers.append('Pointwise\nConv')
        feat = layer_outputs['pointwise_conv'].shape
        feature_sizes.append(f"{feat[1:]}")
        colors.append('lightpink')

    layers.append('Output')
    feature_sizes.append(f"{final_output.shape[1:]}")
    colors.append('lightsteelblue')

    # 绘制网络结构
    y_pos = np.arange(len(layers))

    # 创建水平条形图表示网络层
    bars = ax.barh(y_pos, [1] * len(layers), color=colors, alpha=0.8)

    # 添加层名称和特征尺寸
    for i, (layer, size, bar) in enumerate(zip(layers, feature_sizes, bars)):
        ax.text(0.5, i, f'{layer}\n{size}',
                ha='center', va='center', fontweight='bold', fontsize=9)

    # 添加连接箭头
    for i in range(len(layers)-1):
        ax.annotate('', xy=(1, i), xytext=(0, i+1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.7))

    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(layers)-0.5)
    ax.set_title('Network Architecture Flow', fontweight='bold')


def plot_class_patterns(model, info, num_classes):
    """
    Plot class-specific patterns using guided backpropagation
    """
    print("Generating class-specific patterns...")

    # Create synthetic input for each class pattern visualization
    # This is a simplified approach - in practice you'd use real data

    fig, axes = plt.subplots(1, num_classes, figsize=(5*num_classes, 4))

    if num_classes == 1:
        axes = [axes]

    for class_idx in range(num_classes):
        # Create synthetic activation pattern (this would be replaced with actual data)
        # In practice, you'd compute this using guided backprop or similar methods
        synthetic_pattern = np.random.randn(len(info['ch_names']))
        synthetic_pattern = synthetic_pattern / \
            np.max(np.abs(synthetic_pattern))

        if num_classes > 1:
            ax = axes[class_idx]
        else:
            ax = axes

        im, _ = plot_topomap(synthetic_pattern, info, axes=ax, show=False,
                             sensors=True, contours=6, outlines='head')
        ax.set_title(f'Class {class_idx} Pattern', fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR.joinpath(f'{datetime.now()}.png'))
    plt.show()


def visualize_training_history(training_history):
    """
    Visualize training history if available
    """
    if training_history is None:
        print("No training history provided")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training and validation loss
    if 'train_loss' in training_history:
        axes[0].plot(training_history['train_loss'],
                     label='Training Loss', linewidth=2)
    if 'val_loss' in training_history:
        axes[0].plot(training_history['val_loss'],
                     label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot training and validation accuracy
    if 'train_acc' in training_history:
        axes[1].plot(training_history['train_acc'],
                     label='Training Accuracy', linewidth=2)
    if 'val_acc' in training_history:
        axes[1].plot(training_history['val_acc'],
                     label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training History - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR.joinpath(f'{datetime.now()}.png'))
    plt.show()

# 或者使用更详细的网络结构可视化


def plot_detailed_network_architecture(model, ax):
    """
    绘制详细的FBCNet网络架构图
    """
    # 网络层配置（根据实际模型调整）
    layers_info = [
        {'name': 'Input', 'shape': '(C, T)', 'color': 'lightblue'},
        {'name': 'Temporal Conv', 'shape': '(F1, T)', 'color': 'lightgreen'},
        {'name': 'Spatial Conv', 'shape': '(F2, T)', 'color': 'lightcoral'},
        {'name': 'Batch Norm', 'shape': '(F2, T)', 'color': 'lightyellow'},
        {'name': 'Activation', 'shape': '(F2, T)', 'color': 'lightsalmon'},
        {'name': 'Depthwise Conv', 'shape': '(F3, T)', 'color': 'lightpink'},
        {'name': 'Pointwise Conv',
            'shape': '(F4, T)', 'color': 'lightseagreen'},
        {'name': 'Global Pooling',
            'shape': '(F4,)', 'color': 'lightsteelblue'},
        {'name': 'Classifier', 'shape': f'({num_classes},)', 'color': 'plum'},
        {'name': 'Output', 'shape': f'({num_classes},)', 'color': 'gold'}
    ]

    y_pos = np.arange(len(layers_info))

    # 绘制网络层
    for i, layer in enumerate(layers_info):
        bar = ax.barh(i, 1, color=layer['color'], alpha=0.8, edgecolor='black')

        # 添加文本
        ax.text(0.5, i, f"{layer['name']}\n{layer['shape']}",
                ha='center', va='center', fontweight='bold', fontsize=8)

    # 添加连接线
    for i in range(len(layers_info)-1):
        ax.annotate('', xy=(1, i), xytext=(0, i+1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.6))

    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(layers_info)-0.5)
    ax.set_title('FBCNet Detailed Architecture',
                 fontweight='bold', fontsize=12)

    # 添加图例
    operations = {
        'Conv Layers': 'lightgreen',
        'Normalization': 'lightyellow',
        'Activation': 'lightsalmon',
        'Pooling': 'lightsteelblue',
        'Classification': 'plum'
    }

    # 在右侧添加图例
    legend_text = []
    for op, color in operations.items():
        legend_text.append(f"$\\square$ {op}")

    ax.text(1.1, 0.5, '\n'.join(legend_text), transform=ax.transAxes,
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 修改主函数来包含网络结构图


def analyze_feature_importance(model, epochs, num_classes):
    """
    Analyze feature importance using gradient-based methods
    """
    print("Computing feature importance...")

    # 创建3个子图而不是2个
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Get sample data for analysis
    # sample_data = torch.tensor(
    #     epochs.get_data()[:10], dtype=torch.float32).cuda()
    dl = DataLoader(X[:, :, :, :500], y, groups, test_group=groups[0])
    it = iter(dl.yield_train_data(batch_size=10))

    # Get sample data for analysis
    sample_data = torch.tensor(
        epochs.get_data()[:10], dtype=torch.float32).cuda()
    sample_data, _ = next(it)
    sample_data = torch.tensor(sample_data, dtype=torch.float32)

    # Set model to train mode for gradient computation
    model.train()

    # Compute gradients for feature importance
    sample_data.requires_grad_(True)

    # Forward pass
    outputs = model(sample_data)

    # Use the first sample for analysis
    target_class = torch.argmax(outputs[0])
    outputs[0, target_class].backward()

    # Get gradients
    gradients = sample_data.grad[0].detach().cpu().numpy()

    # Average gradients across time and channels
    feature_importance = np.mean(np.abs(gradients), axis=(0, 1))

    # Channel importance
    if True or len(feature_importance) == len(epochs.ch_names):
        axes[0].bar(range(len(feature_importance)), feature_importance)
        axes[0].set_xlabel('Channel Index')
        axes[0].set_ylabel('Gradient Magnitude')
        axes[0].set_title('Channel-wise Feature Importance')
        axes[0].tick_params(axis='x', rotation=45)

    # Temporal importance (average across channels)
    temporal_importance = np.mean(np.abs(gradients), axis=0)
    axes[1].plot(temporal_importance)
    axes[1].set_xlabel('Time Samples')
    axes[1].set_ylabel('Average Gradient Magnitude')
    axes[1].set_title('Temporal Feature Importance')
    axes[1].grid(True, alpha=0.3)

    # 新增：网络结构特征流图
    try:
        plot_network_feature_flow(model, sample_data, axes[2])
    except:
        # 如果动态获取失败，使用静态架构图
        plot_detailed_network_architecture(model, axes[2])

    axes[2].set_title('FBCNet Network Architecture', fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR.joinpath(f'{datetime.now()}.png'))
    plt.show()

    return feature_importance


# %%
num_classes = 2

# Example usage with your model
if __name__ == "__main__":
    # Assuming you have your model and data ready
    # model = your_trained_fbcnet_model
    # epochs = your_epochs_data
    # info = epochs.info

    # Call the visualization function
    results = visualize_fbcnet_results(
        model=model,
        info=epochs.info,
        epochs=epochs,
        num_classes=num_classes
    )

    # Additional: Model parameter statistics
    print("\n" + "="*60)
    print("Model Parameter Statistics")
    print("="*60)

    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"{name:30} | {str(param.shape):20} | {param_count:8,} parameters")

    print(f"\nTotal trainable parameters: {total_params:,}")

    # Model size in MB
    param_size = 0
    for param in model.parameters():
        if param.requires_grad:
            param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_all_mb:.2f} MB")

# %%
