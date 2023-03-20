import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from keras import models
from PIL import ImageGrab
from PIL import Image

class PaintApp:
    DRAW_WIDTH = 20
    DRAW_COLOR = "black"

    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=560, height=560, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH)

        self.last_x, self.last_y = None, None

        # Bind actions for drawing
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Add clear button
        self.clear_button = tk.Button(root, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.BOTTOM, fill=tk.BOTH)

        # Create the figure and the plot
        self.fig = plt.figure(figsize=(5, 4))
        self.plot = self.fig.add_subplot(111)

        # Add the plot to a FigureCanvasTkAgg widget, which will allow it to be displayed in the Tkinter window
        self.chart = FigureCanvasTkAgg(self.fig, self.root)
        self.chart.draw()
        self.chart.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        
        
        # Load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = models.model_from_json(loaded_model_json)
        # Load weights into new model
        loaded_model.load_weights("model.h5")
        
        self.model=loaded_model

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_canvas(self, event):
        if self.last_x is not None:
            self.canvas.create_line(self.last_x , self.last_y, event.x, event.y,
                                    width=self.DRAW_WIDTH, fill=self.DRAW_COLOR, smooth=1)
        self.last_x, self.last_y = event.x, event.y
        #self.show_charts() # Too slow to work

    def stop_draw(self, event):
        self.last_x, self.last_y = None, None
        self.show_charts()

    def clear(self):
        self.canvas.delete("all")

    def show_charts(self):
        # Clear the existing plot
        self.plot.clear()

        # Predict digit with the model
        labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        data = np.array(self.model.predict(self.canvas_to_nninput()))
        data = data.flatten()
        
        # Console outputs for debugging
        print(data)   
        print("Prediction: ", np.argmax(self.model.predict(self.canvas_to_nninput())))


        # Add the new data to the plot and customize the appearance
        self.plot.bar(labels, data, width=0.5, color='#0000FF')
        self.plot.set_title('Neural network output')
        self.plot.set_xlabel('Probability')
        self.plot.set_ylabel('Digit')

        # Redraw the plot
        self.chart.draw()
    
    def canvas_to_nninput(self):
        
        # Save temporary image
        self.getter()

        # Open the image 
        image = Image.open('tmpImg.jpg')

        # Resize the image to (28, 28)
        image_resized = image.resize((28, 28))

        # Convert the image to grayscale
        image_grayscale = image_resized.convert('L')

        # Convert the image to a NumPy array
        image_preprocessed = np.array(image_grayscale)

        # Normalize the pixel values
        image_normalized = image_preprocessed / 255

        # Reshape the image data to match the input shape of the neural network
        output_data = image_normalized.reshape(1, 28, 28, 1)

        return output_data

    def getter(self):
        # Save canvas to temp image
        widget=self.canvas
        x=root.winfo_rootx()+widget.winfo_x()
        y=root.winfo_rooty()+widget.winfo_y()
        x1=x+widget.winfo_width()
        y1=y+widget.winfo_height()
        ImageGrab.grab().crop((x,y,x1,y1)).save("tmpImg.jpg")

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()