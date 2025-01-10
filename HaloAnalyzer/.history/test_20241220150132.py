import matplotlib.pyplot as plt

# Define the colors
colors = {
    "0": '#FEDA91',
    "1": 'rgb(136,204,238)',
    "2": 'rgb(253,205,172)',
    "3": '#ffbb78',
    "4": '#FF595E',
    "5": '#ff7c43',
    "6": 'rgb(179,205,227)'
}

# Convert RGB strings to hex format
def rgb_to_hex(rgb):
    rgb = rgb.strip('rgb()').split(',')
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

# Convert all colors to hex format
colors = {k: (v if v.startswith('#') else rgb_to_hex(v)) for k, v in colors.items()}

# Plot the color bar
fig, ax = plt.subplots(figsize=(10, 2))

for i, (label, color) in enumerate(colors.items()):
    ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.text(i + 0.5, 0.5, label, ha='center', va='center', fontsize=12, color='black')

ax.set_xlim(0, len(colors))
ax.set_ylim(0, 1)
ax.axis('off')

plt.title('Color Bar')
plt.show()