import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes to the graph
G.add_node('Input Data')
G.add_node('LSTM RNN')
G.add_node('Random Forest')
G.add_node('Gradient Boosting')
G.add_node('Linear Regression')
G.add_node('Output')

# Add edges to the graph
G.add_edge('Input Data', 'LSTM RNN')
G.add_edge('LSTM RNN', 'Random Forest')
G.add_edge('LSTM RNN', 'Gradient Boosting')
G.add_edge('Random Forest', 'Linear Regression')
G.add_edge('Gradient Boosting', 'Linear Regression')
G.add_edge('Linear Regression', 'Output')

# Set the layout and draw the graph
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True)

# Show the plot
plt.show()
