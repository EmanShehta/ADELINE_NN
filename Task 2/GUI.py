from tkinter import *
from NeuralNetwork import *
from tkinter import messagebox
Data = pd.read_csv('..\\penguins.csv')
NumberOfColumns = Data.shape[1]
Columns = Data.columns[1:NumberOfColumns]
Unique_Acutal_Y = Data['species'].unique()
FeacturesList=[]
ClassesList=[]
for i in range(len(Columns)):
    j = i+1
    while(j != len(Columns)):
        FeacturesList.append(Columns[i] + " With " + Columns[j])
        j=j+1

for i in range(len(Unique_Acutal_Y)):
    j = i+1
    while(j != len(Unique_Acutal_Y)):
        ClassesList.append(Unique_Acutal_Y[i] + " With " + Unique_Acutal_Y[j])
        j=j+1
def Run():
    win = Tk()

    '''
        Define Variables 
    '''
    Learning_rate = IntVar()
    Features = StringVar()
    classes = StringVar()
    Epochs = IntVar()
    BiasVAriable = IntVar()
    MSE_threshold = IntVar()

    def show():
        SelectedFeatures = Features.get().split('With')
        SelectedFeatures = list(map(str.strip, SelectedFeatures))
        Selectedclasses = classes.get().split('With')
        Selectedclasses = list(map(str.strip, Selectedclasses))
        # if(len(SelectedFeatures)>0 and len(Selectedclasses)>0 and Epochs.get()>0):
        NN = NeuralNetwork(SelectedFeatures, Selectedclasses, Learning_rate.get(), Epochs.get(), Data,BiasVAriable.get(),MSE_threshold.get())
        Accuracy = NN.Tarin()
        Label(win, text="Accuracy = "+str(Accuracy)).place(x=30, y=150)

        # else:
        #     messagebox.showerror('Error','Error: You have to Entered The Madnadtory Fields')

    win.geometry("800x600")
    win.eval('tk::PlaceWindow . center')

    Features.set("Select two feactures")
    OptionMenu(win, Features, *FeacturesList).place(x=100, y=100)
    # drop.pack()

    classes.set("Select two Calsses")
    OptionMenu(win, classes, *ClassesList).place(x=150, y=150)
    # drop.pack(
    Label(win, text="Learning Rate").place(x=30, y=50)
    Label(win, text="Epochs").place(x=30, y=70)
    Label(win, text="MSE_Threshold").place(x=30, y=20)
    Entry(win, textvariable=Learning_rate).place(x=80, y=50)
    Entry(win, textvariable=Epochs).place(x=80, y=80)
    Entry(win, textvariable=MSE_threshold).place(x=125, y=20)

    Bias_ChickButton = Checkbutton(win, text="Bias", variable=BiasVAriable, onvalue=1, offvalue=0, height=2, width=10)
    Bias_ChickButton.pack()
    button = Button(win, text="click Me", command=show).pack()
    win.mainloop()