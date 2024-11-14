import torch
import time
import argparse
from split_model import get_model_config, initialize_blocks
from torchstat import stat
# from flops_profiler.profiler import get_model_profile
from thop import profile, clever_format
import pandas as pd
import os
from pijuice import PiJuice
import subprocess

pijuice = PiJuice(1, 0x14)

def load_blocks(model_type='resnet50', path='models/cifar10', device=torch.device("cpu")):
    """Load all blocks for the specified model type."""
    # Get model configuration
    ModelClass, block_configs, block_type = get_model_config(model_type)
    
    # Initialize blocks
    blocks = initialize_blocks(ModelClass, block_type, block_configs, model_type, num_classes=10)
    
    # Load saved state dictionaries
    blocks_dir = f'{path}/{model_type}_blocks'
    
    for block_name, block in blocks.items():
        block_path = f"{blocks_dir}/{block_name}.pth"
        state_dict = torch.load(block_path, map_location=device)
        block.load_state_dict(state_dict)
        block.eval()
    
    # Create dummy input
    batch_size = 1
    x = torch.randn(batch_size, 3, 32, 32)
    
    return blocks, x

def measure_block_times(blocks, x, device, repeat_times):
    """Measure execution time for each block."""
    times = {
        "block1": [],
        "block2": [],
        "block3": [],
        "block4": [],
    }

    for i in range(repeat_times):
        print(f"Measuring the {i+1}/{repeat_times} time...")  
        with torch.no_grad():
            # Dummy execution
            # start = time.time()
            out1, _ = blocks['block1'](x)
            # times['dummy'] = time.time() - start

            # print(x.shape)

            # Block 1
            start = time.time()
            out1, _ = blocks['block1'](x)
            times['block1'].append((time.time() - start)*1000)
            
            
            # Block 2
            # print(out1.shape)
            start = time.time()
            out2, _ = blocks['block2'](out1)
            times['block2'].append((time.time() - start)*1000)
            
            # Block 3
            # print(out2.shape)
            start = time.time()
            out3, _ = blocks['block3'](out2)
            times['block3'].append((time.time() - start)*1000)
            
            # Block 4
            # print(out3.shape)
            start = time.time()
            _ = blocks['block4'](out3)
            times['block4'].append((time.time() - start)*1000)
        if i == repeat_times-1:
            break

        time.sleep(60)
    
    return times

def measrue_voltage_and_currency(sps=10, sample_num=100):
    # sps: samples per second
    data = {
        "voltage": [],
        "current": []
    }
    sampling_interval = 1 / sps
    count = 0
    while True:
        # get voltage
        voltage_data = pijuice.status.GetBatteryVoltage()
        battery_voltage = None
        battery_current = None
        if voltage_data['error'] == 'NO_ERROR':
            battery_voltage = voltage_data['data'] / 1000  # V
            print("Battery Voltage:", battery_voltage, "V")
        else:
            print("Error:", voltage_data['error'])

        # get current
        current_data = pijuice.status.GetBatteryCurrent()
        if current_data['error'] == 'NO_ERROR':
            battery_current = current_data['data']  # mA
            print("Battery Current Draw:", battery_current, "mA")
        else:
            print("Error:", current_data['error'])
        
        if battery_voltage is not None and battery_current is not None:
            data['voltage'].append(battery_voltage)
            data['current'].append(battery_current)
        
        count += 1
        if count == sample_num:
            break
        # sampling interval
        time.sleep(sampling_interval)

    return data

def measure_flops(model_name, blocks, block_num, x, device):
    flops_data = {
        "block1": [],
        "block2": [],
        "block3": [],
        "block4": []
    }
    input_sizes = {
        "mobilenetv2": [(1, 3, 32, 32), (1, 24, 8, 8), (1, 32, 4, 4), (1, 64, 2, 2)],
        "resnet18": [(1, 3, 32, 32), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8)],
        "resnet50": [(1, 3, 32, 32), (1, 256, 32, 32), (1, 512, 16, 16), (1, 1024, 8, 8)],
        }
    with torch.no_grad():
        for i in range(block_num):
            model = blocks[f"block{i+1}"]
            mocked_input = torch.zeros(input_sizes[model_name][i]).to(device)
            flops, params = profile(model.to(device), inputs=(mocked_input, ))
            flops, params = clever_format([flops, params], '%.3f')
            flops_data[f"block{i+1}"].append(flops)
            print(flops, params)

    return flops_data


def main():
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Measure execution time for model blocks')
    parser.add_argument('--model', type=str, choices=['resnet50', 'resnet18', 'mobilenetv2'],
                      default='resnet50', help='Model architecture to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                      default='cpu', help='Device to use for computation')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size') 
    parser.add_argument('--path', type=str, default='models/cifar10', help='Path to model blocks')
    parser.add_argument('--host_device', type=str, choices=['rpi2', 'nano'],
                        default='rpi2', help='Device that run the models')
    parser.add_argument('--metrics', type=str, default='flops', choices=['flops', 'proc_time', 'VA', 'VA_trans'], nargs='+', help='Mtrics to measure')
    parser.add_argument('--VA_sps', type=float, default=10, help='sampling rate to collect VA data')
    parser.add_argument('--VA_spnum', type=float, default=200, help='number of samples to collect VA data')
    parser.add_argument('--VA_transrate', type=int, default=100, help='data transmission rate')
    parser.add_argument('--block_num', type=int, default=4, help='Number of blocks to run')
    parser.add_argument('--repeat', type=int, default=50, help='Repeated times for each experiment')

    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load blocks and create input
    # print(f"\nMeasuring execution times for {args.model}")
    # blocks, x = load_blocks(args.model, args.path, device)

    metrics = args.metrics
    block_num = args.block_num
    repeat_times = args.repeat
    model_name = args.model

    data_dir = "measurement_data/rpi2"

    for metric in metrics:
        if metric == 'proc_time':
            #Measure execution times
            times = measure_block_times(blocks, x, device, repeat_times)
            df = pd.DataFrame(times)
            df.to_csv(os.path.join(data_dir, f"proc_time/{model_name}_{repeat_times}.csv"), index=False)
        elif metric == 'flops':
            # Measure flops
            flops = measure_flops(model_name, blocks, block_num, x, device)
            df = pd.DataFrame(flops)
            df.to_csv(os.path.join(data_dir, f"flops/{model_name}.csv"), index=False)
        elif metric == 'VA':
            sps = args.VA_sps
            sample_num = args.VA_spnum
            va_data = measrue_voltage_and_currency(sps, sample_num)
            df = pd.DataFrame(va_data)
            df.to_csv(os.path.join(data_dir, f"VAs/{model_name}_{sps}_{sample_num}.csv"), index=False)
        elif metric == 'VA_trans':
            sps = args.VA_sps
            sample_num = args.VA_spnum
            trans_rate = args.VA_transrate
            process = subprocess.Popen(['python', 'run_transmission_sender.py', '--rate', str(trans_rate)])
            time.sleep(5)
            va_data = measrue_voltage_and_currency(sps, sample_num)
            df = pd.DataFrame(va_data)
            df.to_csv(os.path.join(data_dir, f"VAs-trans/{trans_rate}_{sps}_{sample_num}.csv"), index=False)
            process.terminate()
        else:
            print(f"Unknown metric {metric}!")

if __name__ == "__main__":
    main()
