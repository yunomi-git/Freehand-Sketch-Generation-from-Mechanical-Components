import cairosvg

# Test with minimal SVG
test_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'
file = "out/class_0/CAD_MODEL_A/class_0==CAD_MODEL_A_0.svg"
with open(file, 'r') as f:
    svg = f.read()
# cairosvg.svg2png(bytestring=svg, write_to='test.png')
cairosvg.svg2png(url=file, write_to="out/class_0/CAD_MODEL_A/class_0==CAD_MODEL_A_2.png", output_width=512, output_height=512)