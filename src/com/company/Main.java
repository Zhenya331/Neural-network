package com.company;

import Activation.Tanh;
import Model.Model;
import Model.ModelActions;
import Model.Loss;
import Model.Optimizer;
import Activation.ReLu;
import Activation.SoftMax;
import Activation.Sigmoid;
import Model.Log;

import java.util.Arrays;

public class Main {

    public static void PrintMassive(double[][][][] massive, int[] size) {
        System.out.println("Weights Conv2D:");
        for (int i = 0; i < size[0]; i++) {
            for (int j = 0; j < size[1]; j++) {
                for (int k = 0; k < size[2]; k++) {
                    for (int l = 0; l < size[3]; l++) {
                        System.out.print(massive[i][j][k][l] + "\t");
                    }
                    System.out.println();
                }
                System.out.println("////////////////////////////////////////////////////");
            }
            System.out.println("--------------------------------------------------------");
        }
        System.out.println("End Print Weights Conv2D");
    }
    public static void PrintMassive(double[][][] massive, int[] size) {
        System.out.println("Neurons Conv2D:");
        for (int i = 0; i < size[0]; i++) {
            for (int j = 0; j < size[1]; j++) {
                for (int k = 0; k < size[2]; k++) {
                    System.out.print(massive[i][j][k] + "\t");
                }
                System.out.println();
            }
            System.out.println("--------------------------------------------------------");
        }
        System.out.println("End Print Neurons Conv2D");
    }
    public static void PrintMassive(double[][] massive, int[] size) {
        System.out.println("Weights Dense:");
        for (int i = 0; i < size[0]; i++) {
            for (int j = 0; j < size[1]; j++) {
                System.out.print(massive[i][j] + "\t");
            }
            System.out.println();
        }
        System.out.println("End Print Weights Dense");
    }
    public static void PrintMassive(double[] massive, int[] size) {
        System.out.println("Neurons Dense:");
        for (int i = 0; i < size[0]; i++) {
            System.out.print(massive[i] + "\t");
        }
        System.out.println();
        System.out.println("End Print Neurons Dense");
    }

    public static void main(String[] args) throws CloneNotSupportedException {

        Model model = new Model(28, 1);
        model.Conv2DCreate(3, 30, 1, new ReLu(), 0.0);
        model.MaxPoolingCreate(2);
        model.Conv2DCreate(3, 15, 1, new ReLu(), 0.0);
        model.MaxPoolingCreate(2);
        model.DenseCreate(10,new SoftMax(), 0.1);
        model.InitializeWeights(-0.002, 0.05);

        ModelActions modelActions = new ModelActions(model);

        ForDataset dataTrain = FileWorkMNIST.GetDataset("train");
        ForDataset dataTest = FileWorkMNIST.GetDataset("test");
        modelActions.Fit(dataTrain, dataTest, 200, 5, Loss::mse, Optimizer::BackPropagation, 0.0001);

        //modelActions.Evaluate(dataTest);
    }
}
