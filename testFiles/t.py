from graphviz import Digraph

# create a new graph
dot = Digraph()

# add nodes
dot.node('Input', shape='parallelogram')
dot.node('Preprocessing', shape='diamond')
dot.node('LSTM', shape='box')
dot.node('RandomForest', shape='box')
dot.node('GradientBoosting', shape='box')
dot.node('LinearRegression', shape='box')
dot.node('Output', shape='parallelogram')

# add edges
dot.edge('Input', 'Preprocessing')
dot.edge('Preprocessing', 'LSTM')
dot.edge('Preprocessing', 'RandomForest')
dot.edge('Preprocessing', 'GradientBoosting')
dot.edge('Preprocessing', 'LinearRegression')
dot.edge('LSTM', 'Output')
dot.edge('RandomForest', 'Output')
dot.edge('GradientBoosting', 'Output')
dot.edge('LinearRegression', 'Output')

# save the graph as a PNG file
dot.render('system_architecture', format='png')
