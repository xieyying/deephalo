# Define the color palettes for each image
color_palettes = {
    "science.adr6006-f1.jpg": [
        "rgb(19, 18, 18)",
        "rgb(206, 214, 218)",
        "rgb(245, 231, 216)",
        "rgb(233, 240, 244)",
        "rgb(124, 178, 208)",
        "rgb(77, 72, 79)",
        "rgb(119, 122, 140)",
        "rgb(37, 119, 146)",
        "rgb(182, 191, 203)",
        "rgb(243, 201, 163)",
        "rgb(153, 173, 161)",
        "rgb(223, 133, 70)"
    ],
    "science.adr6006-f2.jpg": [
        "rgb(13, 12, 12)",
        "rgb(224, 227, 227)",
        "rgb(203, 224, 210)",
        "rgb(226, 230, 194)",
        "rgb(197, 219, 239)",
        "rgb(245, 247, 213)",
        "rgb(155, 191, 226)",
        "rgb(186, 196, 181)",
        "rgb(63, 64, 65)",
        "rgb(113, 116, 114)",
        "rgb(231, 240, 241)",
        "rgb(149, 165, 159)",
        "rgb(224, 212, 93)",
        "rgb(93, 152, 190)",
        "rgb(251, 222, 221)",
        "rgb(247, 190, 157)",
        "rgb(220, 123, 75)"
    ],
    "science.adr6006-f4.jpg": [
        "rgb(167, 194, 226)",
        "rgb(203, 225, 233)",
        "rgb(22, 22, 23)",
        "rgb(251, 252, 251)",
        "rgb(176, 179, 182)",
        "rgb(227, 241, 244)",
        "rgb(85, 90, 102)",
        "rgb(193, 230, 250)",
        "rgb(141, 145, 151)",
        "rgb(88, 127, 180)",
        "rgb(228, 230, 233)",
        "rgb(206, 209, 212)",
        "rgb(246, 247, 248)",
        "rgb(251, 213, 168)",
        "rgb(238, 163, 73)",
        "rgb(244, 239, 236)",
        "rgb(249, 215, 207)",
        "rgb(222, 187, 183)"
    ]
}

# Print the color palettes
for image, colors in color_palettes.items():
    print(f"Color palette for {image}:")
    for color in colors:
        print(color)
    print()