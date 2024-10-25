import numpy as np
from rich.console import Console
from rich.table import Table

from main_spiking_rml18 import main

GPU_ID = 1
ADC_RESOLUTIONS = [4]
TIMESTEPS_PER_SPIKE = [512, 256, 128, 64, 32, 16]
EPOCHS = 20


if __name__ == '__main__':
    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ADC bits")
    table.add_column("Samples per Frame")
    table.add_column("Run name")
    table.add_column("Average Accuracy (%)")

    accuracies = np.zeros((len(ADC_RESOLUTIONS), len(TIMESTEPS_PER_SPIKE)))
    for i, adc_resolution in enumerate(ADC_RESOLUTIONS):
        for j, timesteps_per_spike in enumerate(TIMESTEPS_PER_SPIKE):
            run_name, accuracies[i, j] = main(
                gpu_id=GPU_ID,
                epochs=EPOCHS,
                adc_resolution=adc_resolution,
                timesteps_per_spike=timesteps_per_spike,
                train=False
            )

            table.add_row(
                str(adc_resolution),
                str(timesteps_per_spike),
                run_name,
                f"{accuracies[i,j]*100:.2f}"
            )
            print()

    console.print(table)
