# Import the necessary library
from sage.graphs.graph_generators import graphs

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Create a PDF with the plots and text
with PdfPages('graphs_and_text.pdf') as pdf:
    # First page with the complete graph
    G1 = graphs.CompleteGraph(5)
    plot1 = G1.plot(vertex_size=300, vertex_labels=True, edge_color='blue')
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.9, 'Here is a complete graph on 5 vertices:', ha='center', fontsize=12)
    
    # Convert Sage plot to a Matplotlib-compatible image
    plot1_matplotlib = plot1.matplotlib()
    plot1_matplotlib.savefig('temp_plot1.png', bbox_inches='tight')
    img1 = plt.imread('temp_plot1.png')
    plt.imshow(img1)
    plt.axis('off')
    pdf.savefig()
    plt.close()
    
    # Second page with the Petersen graph
    G2 = graphs.PetersenGraph()
    G2.set_pos(G2.layout('spring'))
    plot2 = G2.plot(vertex_size=500, vertex_labels=True, edge_color='green', vertex_color='red')
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.9, 'Here is a Petersen graph:', ha='center', fontsize=12)
    
    # Convert Sage plot to a Matplotlib-compatible image
    plot2_matplotlib = plot2.matplotlib()
    plot2_matplotlib.savefig('temp_plot2.png', bbox_inches='tight')
    img2 = plt.imread('temp_plot2.png')
    plt.imshow(img2)
    plt.axis('off')
    pdf.savefig()
    plt.close()

# Clean up temporary files
import os
os.remove('temp_plot1.png')
os.remove('temp_plot2.png')