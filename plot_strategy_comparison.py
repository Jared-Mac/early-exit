import matplotlib.pyplot as plt
import numpy as np

strategies = ['DQL', 'ALWAYS_TRANSMIT', 'EARLY_EXIT']
accuracy = [0.77, 0.77, 0.75]
latency = [0.03, 0.15, 0.06]
avg_data_sent = [239206.40, 262144.00, 0.00]
avg_cpu_flops = [562000000.00, 200000000.00, 1223600000.00]

x = np.arange(len(strategies))
width = 0.2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

rects1 = ax1.bar(x - width*1.5, accuracy, width, label='Accuracy', color='b', alpha=0.7)
rects2 = ax1.bar(x - width/2, latency, width, label='Avg Latency (s)', color='g', alpha=0.7)
rects3 = ax1.bar(x + width/2, [d/1e6 for d in avg_data_sent], width, label='Avg Data Sent (MB)', color='r', alpha=0.7)
rects4 = ax1.bar(x + width*1.5, [f/1e9 for f in avg_cpu_flops], width, label='Avg CPU GFLOPS', color='purple', alpha=0.7)

ax1.set_ylabel('Values')
ax1.set_title('Comparison of DQL, ALWAYS_TRANSMIT, and EARLY_EXIT Strategies')
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.15), ncol=2)

# Exit numbers
exit_numbers = {
    'DQL': [1158, 399, 158, 27],
    'ALWAYS_TRANSMIT': [2000, 0, 0, 0],
    'EARLY_EXIT': [784, 711, 228, 277]
}

bottom = np.zeros(3)
for i, block in enumerate(['block0', 'block1', 'block2', 'block3']):
    values = [exit_numbers[s][i] for s in strategies]
    ax2.bar(x, values, width, bottom=bottom, label=f'Exit {block}')
    bottom += values

ax2.set_ylabel('Exit Numbers')
ax2.set_xlabel('Strategies')
ax2.legend(loc='upper right')

plt.xticks(x, strategies)
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')

# Optionally, you can still display the plot
# plt.show()
