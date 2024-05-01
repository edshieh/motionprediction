import argparse
import os
import re
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir")
    parser.add_argument("--epochs", type=int)

    args = parser.parse_args()

    for i, filename in enumerate(os.listdir(args.dir)):
        file_path = os.path.join(args.dir, filename)
        if os.path.isfile(file_path) and file_path.split(".")[1] in ("txt",):
            with open(file_path) as f:
                print(file_path)
                file_input = f.read()
                train = re.findall(r"Training loss (\d*\.?\d*)", file_input)
                val = re.findall(r"Validation loss (\d*\.?\d*)", file_input)
                plt.figure()
                plt.plot(list(range(1, args.epochs+1)), [4*round(float(t),4) for t in train[:args.epochs]], label='Training')
                plt.plot(list(range(1, args.epochs+1)), [4*round(float(t),4) for t in val[:args.epochs]], label='Validation')
                plt.legend()
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title(filename.split(".")[0])

                # Save the plot to a file
                plt.savefig(f'plot{i}.png')