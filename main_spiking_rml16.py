import os
import sys

import numpy as np
import torch
import torch.utils.data

import torch_framework.spiking as spiking
from rml16_loader import RML16Loader

DATAFILE = "/data/shared/datasets/rf_ml/RML2016_10A/RML2016.10a_dict.pkl"
DATASET_SIZE = 220000
SNR_LIST = np.arange(-20, 19, 2)

SEED = 42
GPU_ID = 1
TRAIN_VAL_TEST_SPLIT = [0.6, 0.2, 0.2]
BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

ADC_RESOLUTION = 4
TIMESTEPS_PER_SPIKE = 16
C = 3.75
LAMBDA = 0.25
THRESHOLD = 0.25
ALPHA = 0.5

TRAIN = True

def main(**kwargs):
    datafile = kwargs.get("datafile", DATAFILE)
    dataset_size = kwargs.get("dataset_size", DATASET_SIZE)
    snr_list = kwargs.get("snr_list", SNR_LIST)
    seed = kwargs.get("seed", SEED)
    gpu_id = kwargs.get("gpu_id", GPU_ID)
    train_val_test_split = kwargs.get("train_val_test_split", TRAIN_VAL_TEST_SPLIT)
    batch_size = kwargs.get("batch_size", BATCH_SIZE)
    epochs = kwargs.get("epochs", EPOCHS)
    learning_rate = kwargs.get("learning_rate", LEARNING_RATE)
    weight_decay = kwargs.get("weight_decay", WEIGHT_DECAY)
    adc_resolution = kwargs.get("adc_resolution", ADC_RESOLUTION)
    timesteps_per_spike = kwargs.get("timesteps_per_spike", TIMESTEPS_PER_SPIKE)
    c = kwargs.get("c", C)
    lambd = kwargs.get("lambd", LAMBDA)
    threshold = kwargs.get("threshold", THRESHOLD)
    alpha = kwargs.get("alpha", ALPHA)
    scaling = 1 / (c + lambd)
    mult_factor = (c - lambd) / (c + lambd)
    train_model = kwargs.get("train", TRAIN)

    run_name = f"rml16_constellation_spiking_norm_{adc_resolution}_{timesteps_per_spike}_{epochs}_{snr_list[0]}_{snr_list[-1]}_{seed}"
    print(f"Run name: {run_name}")

    def transform(x):
        x = np.transpose(x)
        x = x / np.sqrt(np.mean(x**2))
        x = np.clip(x, -4, 4)
        x = (x + 4) / 8
        x = x.astype(np.float32)
        x = np.floor(x * (2 ** adc_resolution - 1)).astype(np.int32)
        # shape: (128, 2)
        constellation_map = np.zeros((1, 2 ** adc_resolution, 2 ** adc_resolution, x.shape[0] // timesteps_per_spike), dtype=np.float32)
        x = np.concatenate([x, np.arange(x.shape[0])[:, None] // timesteps_per_spike], axis=-1)
        constellation_map[0, x[:, 0], x[:, 1], x[:, 2]] = 1
        return constellation_map

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create the device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # SNRs of each datapoint
    snrs = []
    for _ in range(11):
        for snr in snr_list:
            snrs.append(np.zeros(1000) + snr)
    snrs = np.concatenate(snrs)

    indices = np.random.permutation(dataset_size)
    train_indices = indices[:int(train_val_test_split[0] * dataset_size)]
    val_indices = indices[int(train_val_test_split[0] * dataset_size):int(train_val_test_split[0] * dataset_size) + int(train_val_test_split[1] * dataset_size)]
    test_indices = indices[int(train_val_test_split[0] * dataset_size) + int(train_val_test_split[1] * dataset_size):]
    test_indices_snrs = {}
    for snr in snr_list:
        test_indices_snrs[str(snr)] = test_indices[np.where(snrs[test_indices] == snr)[0]]

    # Load the data
    train_dataset = RML16Loader(
        path=datafile,
        indices=train_indices,
        cache=True,
        cache_file=f"cache/rml16_train_{seed}",
        transform=transform)

    val_dataset = RML16Loader(
        path=datafile,
        indices=val_indices,
        cache=True,
        cache_file=f"cache/rml16_val_{seed}",
        transform=transform)

    test_datasets = [
        RML16Loader(
            path=datafile,
            indices=test_indices_snrs[str(snr)],
            cache=True,
            cache_file=f"cache/rml16_test_{snr}_{seed}",
            transform=transform)
        for snr in snr_list
    ]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset sizes: {[len(test_dataset) for test_dataset in test_datasets]}")

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    test_loaders = [
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        for test_dataset in test_datasets
    ]

    # Create the model
    model = spiking.network.Sequential(
        (1, 2 ** adc_resolution, 2 ** adc_resolution, 1),
        scaling=scaling,
        mult_factor=mult_factor,
        threshold=threshold,
        alpha=alpha)
    model.add("conv2d", 32)
    model.add("maxpool2d")
    model.add("dropout", p=0.3)
    model.add("conv2d", 64)
    model.add("maxpool2d")
    model.add("dropout", p=0.4)
    model.add("flatten")
    model.add("linear", 128)
    model.add("dropout", p=0.5)
    model.add("output", 11)
    model.to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    def train_step(model, data, target, criterion, optimizer):
        model.train()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return loss.item(), correct

    def val_step(model, data, target, criterion):
        model.eval()
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
        return loss.item(), correct

    # Training loop
    os.makedirs("saved_models", exist_ok=True)
    savepath = os.path.join("saved_models", f"{run_name}.pt")
    if train_model:
        best_val_loss = float("inf")
        for e in range(epochs):
            # Train
            train_loss = 0.0
            train_correct = 0.0
            for i, (data, target) in enumerate(train_loader):
                loss, correct = train_step(model, data, target, loss_fn, optimizer)
                train_loss += loss
                train_correct += correct
                print_string = f"Epoch {e + 1}/{epochs} - Batch {i + 1}/{len(train_loader)} - " \
                                f"Train Loss: {train_loss / ((i+1)*len(data)):.4f}, " \
                                f"Train Accuracy: {train_correct / ((i+1)*len(data)):.4f}"
                sys.stdout.write(f"\r\033[K{print_string}")
                sys.stdout.flush()
            train_loss /= len(train_loader.dataset)
            train_correct /= len(train_loader.dataset)
            print_string = f"Epoch {e + 1}/{epochs} - Batch {len(train_loader)}/{len(train_loader)} - " \
                            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_correct:.4f}"
            sys.stdout.write(f"\r\033[K{print_string}")
            sys.stdout.flush()

            # Validate
            val_loss = 0.0
            val_correct = 0.0
            for data, target in val_loader:
                loss, correct = val_step(model, data, target, loss_fn)
                val_loss += loss
                val_correct += correct
            val_loss /= len(val_loader.dataset)
            val_correct /= len(val_loader.dataset)

            print_string += f" - Val Loss: {val_loss:.4f}, Val Accuracy: {val_correct:.4f}"
            sys.stdout.write(f"\r\033[K{print_string}\n")
            sys.stdout.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), savepath)

            lr_scheduler.step()

    model.load_state_dict(torch.load(savepath))
    # Test the model
    test_losses = []
    test_corrects = []
    for test_loader in test_loaders:
        test_loss = 0.0
        test_correct = 0.0
        for data, target in test_loader:
            loss, correct = val_step(model, data, target, loss_fn)
            test_loss += loss
            test_correct += correct
        test_loss /= len(test_loader.dataset)
        test_correct /= len(test_loader.dataset)
        test_losses.append(test_loss)
        test_corrects.append(test_correct)

    for snr, test_loss, test_correct in zip(snr_list, test_losses, test_corrects):
        print(f"SNR: {snr} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_correct:.4f}")
    print(f"Average accuracy = {np.mean(np.array(test_corrects)):.4f}")

    os.makedirs("results", exist_ok=True)
    np.save(f"results/{run_name}.npy", np.array([snr_list, test_losses, test_corrects]))

    print(f"Average accuracy = {np.mean(np.array(test_corrects)):.4f}")

    return run_name, np.mean(np.array(test_corrects))


if __name__ == '__main__':
    main()
