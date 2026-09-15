import re
import matplotlib.pyplot as plt

def extract_losses(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    epoch_losses = {}
    validation_losses = []
    actual_errors = []  
    val_pattern = re.compile(r'\[Epoch (\d+)\] Mean L1 validation error: ([\d\.]+) hours.')
    train_pattern = re.compile(r'Iteration \d+, train loss = ([\d\.]+) \| Epoch (\d+)')
    
    for line in lines:
        val_match = val_pattern.match(line)
        if val_match:
            epoch, actual_error = map(float, val_match.groups())
            validation_losses.append((int(epoch), actual_error))
            actual_errors.append((int(epoch), actual_error))
        
        train_match = train_pattern.match(line)
        if train_match:
            train_loss, epoch = float(train_match.group(1)), int(train_match.group(2))
            if epoch not in epoch_losses:
                epoch_losses[epoch] = []
            epoch_losses[epoch].append(train_loss)
    
    
    # Prepare data for plotting
    epochs = sorted(epoch_losses.keys())
    avg_train_losses = [sum(epoch_losses[e]) / len(epoch_losses[e]) for e in epochs]
    val_epochs, actual_vals = zip(*validation_losses) if validation_losses else ([], [], [])
    
    # Plot loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, avg_train_losses, label='Train Loss', marker='o')
    plt.plot(val_epochs, actual_vals, label='Mean Actual Error', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Error')
    plt.title('Train and Validation Loss per Epoch')
    plt.legend()
    plt.grid()
    plt.savefig("curve4_14.png")
    plt.close()

# 使用示例
extract_losses('output_4_14.txt', 'extract.txt')
