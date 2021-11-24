from PyQt5 import QtCore, QtGui, QtWidgets
from Melanoma.MelanomaClassifier import MelanomaClassifier
from Melanoma.model_evaluation import ObtainModelThreshold
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


class Ui_Main(QtWidgets.QWidget):

    def setupUi(self, Main):
        Main.setObjectName("Melanoma Diagnostic Tool")

        # Create default variables
        self.MelanomaImg_Path = "./"
        self.AI_Model = None

        # Create stack of windows
        self.QtStack = QtWidgets.QStackedLayout()

        # Create main stack layers and stack layout
        self.stack_Main = QtWidgets.QWidget()
        self.MainMenuUI()

        # Create stack
        self.QtStack.insertWidget(0, self.stack_Main)
        # Add here additional layers

        # Create connections
        self.PushButton1.clicked.connect(self.loadMelanomaImage)
        self.PushButton2.clicked.connect(self.runModel)

    # Define Main screen
    def MainMenuUI(self):
        self.stack_Main.resize(480, 750)
        self.stack_Main.setStyleSheet("background-color: rgb(255, 255, 255);")

    # Title for the GUI #
        self.title = QtWidgets.QLabel(self.stack_Main)
        self.title.setGeometry(QtCore.QRect(20, 10, 470, 45))
        self.title.setText("Melanoma Diagnostic Tool")
        self.title.setStyleSheet("font: 35pt \"Arial Rounded MT Bold\";\n"
                                 "color: rgb(46, 117, 182);\n"
                                 "text-align:center;\n"
                                 "\n"
                                 "\n"
                                 "")

    #Text Box for path #
        self.PathText = QtWidgets.QLineEdit(self.stack_Main)
        self.PathText.setGeometry(QtCore.QRect(10, 70, 240, 80))
        self.PathText.setStyleSheet("background-color: rgba(157, 195, 230, 127);\n"
                                    "font: 15pt \"Arial Rounded MT Bold\";\n"
                                    "border-radius: 15px;\n"
                                    "color: rgb(46, 117, 182);\n"
                                    "\n"
                                    "\n"
                                    "")
        self.PathText.setText(self.MelanomaImg_Path)

    #PushButton1#
        self.PushButton1 = QtWidgets.QPushButton(self.stack_Main)
        self.PushButton1.setGeometry(QtCore.QRect(270, 70, 200, 80))
        self.PushButton1.setStyleSheet("background-color: rgb(157, 195, 230);\n"
                                       "font: 20pt \"Arial Rounded MT Bold\";\n"
                                       "border-radius: 15px;\n"
                                       "color: rgb(46, 117, 182);\n"
                                       "\n"
                                       "\n"
                                       "")
        self.PushButton1.setAutoDefault(False)
        self.PushButton1.setDefault(True)
        self.PushButton1.setFlat(False)
        self.PushButton1.setObjectName("load_image")
        self.PushButton1.setText("Load Image")

    #Melanoma Image#
        self.MelanomaImg = QtWidgets.QLabel(self.stack_Main)
        self.Img = QtGui.QPixmap('GUI/ExampleMark.png').scaled(255, 255)
        self.MelanomaImg.setPixmap(self.Img)
        self.MelanomaImg.setGeometry(QtCore.QRect(105, 190, 255, 255))
        self.MelanomaImg.setStyleSheet("background-color: rgb(157, 195, 230);\n"
                                       "font: 20pt \"Arial Rounded MT Bold\";\n"
                                       "border-radius: 15px;\n"
                                       "color: rgb(46, 117, 182);\n"
                                       "\n"
                                       "\n"
                                       "")
        self.MelanomaImg.setObjectName("melanomaImg")

    #PushButton2#
        self.PushButton2 = QtWidgets.QPushButton(self.stack_Main)
        self.PushButton2.setGeometry(QtCore.QRect(10, 480, 460, 181))
        self.PushButton2.setStyleSheet("background-color: rgb(157, 195, 230);\n"
                                       "font: 35pt \"Arial Rounded MT Bold\";\n"
                                       "border-radius: 15px;\n"
                                       "color: rgb(46, 117, 182);\n"
                                       "\n"
                                       "\n"
                                       "")
        self.PushButton2.setText("Launch AI Diagnostics")

    # Diagnosis Result #
        self.ResultText = QtWidgets.QLabel(self.stack_Main)
        self.ResultText.setGeometry(QtCore.QRect(15, 680, 470, 45))
        self.ResultText.setText("Diagnosis Result: ")
        self.ResultText.setStyleSheet("font: 35pt \"Arial Rounded MT Bold\";\n"
                                      "color: rgb(46, 117, 182);\n"
                                      "text-align:center;\n"
                                      "\n"
                                      "\n"
                                      "")

    # Unknown Result #
        self.UnknownText = QtWidgets.QLabel(self.stack_Main)
        self.UnknownText.setGeometry(QtCore.QRect(310, 680, 470, 45))
        self.UnknownText.setText("Unknown")
        self.UnknownText.setStyleSheet("font: 35pt \"Arial Rounded MT Bold\";\n"
                                       "color: rgb(46, 117, 182);\n"
                                       "text-align:center;\n"
                                       "\n"
                                       "\n"
                                       "")
        self.UnknownText.setVisible(True)

    # Positive Result #
        self.PositiveText = QtWidgets.QLabel(self.stack_Main)
        self.PositiveText.setGeometry(QtCore.QRect(310, 680, 470, 45))
        self.PositiveText.setText("Positive")
        self.PositiveText.setStyleSheet("font: 35pt \"Arial Rounded MT Bold\";\n"
                                        "color: rgb(255, 85, 85);\n"
                                        "text-align:center;\n"
                                        "\n"
                                        "\n"
                                        "")
        self.PositiveText.setVisible(False)

    # Negative Result #
        self.NegativeText = QtWidgets.QLabel(self.stack_Main)
        self.NegativeText.setGeometry(QtCore.QRect(310, 680, 470, 45))
        self.NegativeText.setText("Negative")
        self.NegativeText.setStyleSheet("font: 35pt \"Arial Rounded MT Bold\";\n"
                                        "color: rgb(135, 222, 135);\n"
                                        "text-align:center;\n"
                                        "\n"
                                        "\n"
                                        "")
        self.NegativeText.setVisible(False)

    def loadMelanomaImage(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Melanoma Image to analyze...", "", "All Files (*);;PNG Files (*.png);;JPEG Files (*.jpeg, *.jpg)", options=options)
        if fileName:

            # Check if image is .png / .jpeg / .jpg
            validExtensions = ['jpg', 'jpeg', 'jpg']
            if fileName.split(".")[-1] in validExtensions:
                # Store path
                self.MelanomaImg_Path = fileName
                # Load path on text box
                self.PathText.setText(self.MelanomaImg_Path)
                # Display image in IMG square
                self.Img = QtGui.QPixmap(
                    self.MelanomaImg_Path).scaled(255, 255)
                self.MelanomaImg.setPixmap(self.Img)

    def runModel(self):

        # Check if image is loaded and path is valid
        validExtensions = ['jpg', 'jpeg', 'jpg']
        if (not self.MelanomaImg_Path.split(".")[-1] in validExtensions) or (not os.path.exists(self.MelanomaImg_Path)):
            return

        # Update push button text
        self.PushButton2.setText("Running Diagnostics...")
        self.PushButton2.update()

        # Load model and run it
        if self.AI_Model == None:
            self.AI_Model = MelanomaClassifier(
                "Models/Full Model/EFN_BestAUC/")
        self.targetImg = plt.imread(self.MelanomaImg_Path)
        Predict_Proba = self.AI_Model.predict(self.targetImg[np.newaxis, :])

        # Load threshold from file
        f = open("Models/BestThres_EFN.txt", "r")
        thres = float(f.readline())
        f.close()

        # Print debug info
        print("Prediction Proba -> " + str(Predict_Proba))
        print("Threshold -> " + str(thres))

        # Create prediction
        if Predict_Proba > thres:
            # Positive
            self.UnknownText.setVisible(False)
            self.NegativeText.setVisible(False)
            self.PositiveText.setVisible(True)
        else:
            # Negative
            self.UnknownText.setVisible(False)
            self.NegativeText.setVisible(True)
            self.PositiveText.setVisible(False)

        self.PushButton2.setText("Launch AI Diagnostics")
