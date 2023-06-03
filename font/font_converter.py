import fontforge
F = fontforge.open("font/DIN 1451 Mittelschrift Regular.ttf")
for name in F:
    filename = name + ".png"
    # print name
    F[name].export(filename)