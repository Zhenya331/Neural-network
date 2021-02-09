package Model;

import Activation.ForMaxPooling;
import Activation.IActivation;

public class Layer implements Cloneable {
    public double[] data;
    public String type;
    public int[] sizeData;
    public IActivation activation;
    public int[][][] indexMaxPooling;
    public int stride;

    // Convolution layer
    public Layer(int SizeOfImage, int NumberOfImages, IActivation Activation) {
        data = new double[SizeOfImage * SizeOfImage * NumberOfImages];
        type = TypeOfLayer.Conv2D;
        sizeData = new int[] {NumberOfImages, SizeOfImage, SizeOfImage};
        activation = Activation;
        //System.out.println("Conv2D layer: " + sizeData[0] + "*" + sizeData[1] + "*" + sizeData[2]);
    }

    // Dense layer
    public Layer(int NumberOfNeurons, IActivation Activation) {
        data = new double[NumberOfNeurons];
        type = TypeOfLayer.Dense;
        sizeData = new int[] {NumberOfNeurons};
        activation = Activation;
        //System.out.println("Dense layer: " + sizeData[0]);
    }

    // MaxPooling layer
    public Layer (int stride, int[] sizeInputData) {
        type = TypeOfLayer.MaxPooling;
        sizeData = new int[] {sizeInputData[0], sizeInputData[1] / stride, sizeInputData[2] / stride};
        data = new double[sizeData[0] * sizeData[1] * sizeData[2]];
        indexMaxPooling = new int[sizeData[0]][sizeData[1] * sizeData[2]][2];
        this.stride = stride;
        activation = new ForMaxPooling();
    }

    public double[][][] ParseDataConv2D() {
        if (type.equals(TypeOfLayer.Conv2D) || type.equals(TypeOfLayer.MaxPooling)) {
            double[][][] result = new double[sizeData[0]][sizeData[1]][sizeData[2]];
            int count = 0;
            for(int i = 0; i < sizeData[0]; i++) {
                for(int j = 0; j < sizeData[1]; j++) {
                    for(int k = 0; k < sizeData[2]; k++) {
                        result[i][j][k] = data[count];
                        count++;
                    }
                }
            }
            return result;
        }
        System.out.println("Неправильный тип уровня");
        return null;
    }

    public void FillDataConv2D(double[][][] Data) {
        if (type.equals(TypeOfLayer.Conv2D) || type.equals(TypeOfLayer.MaxPooling)) {
            int count = 0;
            for(int i = 0; i < sizeData[0]; i++) {
                for(int j = 0; j < sizeData[1]; j++) {
                    for(int k = 0; k < sizeData[2]; k++) {
                        data[count] = Data[i][j][k];
                        count++;
                    }
                }
            }
        } else {
            System.out.println("Неправильный тип уровня");
        }
    }

    public void FillDataDense(double[] Data) {
        if (type.equals(TypeOfLayer.Dense)) {
            System.arraycopy(Data, 0, data, 0, Data.length);
        } else {
            System.out.println("Неправильный тип уровня");
        }
    }

    public Layer clone() throws CloneNotSupportedException{
        return (Layer) super.clone();
    }
}