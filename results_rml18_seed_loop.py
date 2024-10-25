import numpy as np
from rich.console import Console
from rich.table import Table

from main_spiking_rml18 import main

GPU_ID = 0
ADC_RESOLUTION = 4
TIMESTEPS_PER_SPIKE = 16
EPOCHS = 50


if __name__ == '__main__':
    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ADC bits")
    table.add_column("Samples per Frame")
    table.add_column("Seed")
    table.add_column("Run name")
    table.add_column("Average Accuracy (%)")

    accuracies = np.zeros(10)
    for i, seed in enumerate(range(10)):
        run_name, accuracies[i] = main(
            gpu_id=GPU_ID,
            epochs=EPOCHS,
            seed=seed,
            adc_resolution=ADC_RESOLUTION,
            timesteps_per_spike=TIMESTEPS_PER_SPIKE,
            train=True
        )

        table.add_row(
            str(ADC_RESOLUTION),
            str(TIMESTEPS_PER_SPIKE),
            str(seed),
            run_name,
            f"{accuracies[i]*100:.2f}"
        )
        print()

    console.print(table)
    print(f"Average accuracy across runs = {accuracies.mean()*100:.2f}%")
