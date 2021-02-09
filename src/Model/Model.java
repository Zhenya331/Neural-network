package Model;

import java.util.ArrayList;
import Activation.IActivation;


public class Model implements Cloneable {
    public ArrayList<Layer> Layers;
    public ArrayList<Weight> Weights;
    public int NumberOfLayers;
    public double[][][] InputForm;

    public Model(int SizeOfImage, int NumberOfImages) {
        InputForm = new double[NumberOfImages][SizeOfImage][SizeOfImage];
        NumberOfLayers = 0;
        Layers = new ArrayList<>();
        Weights = new ArrayList<>();
    }

    public Model() {

    }

    public void InitializeWeights(double LowerLimit, double UpperLimit) {
        if(UpperLimit > LowerLimit) {
            for (Weight weight: Weights) {
                if (!weight.type.equals(TypeOfLayer.MaxPooling)) {
                    for (int i = 0; i < weight.data.length; i++) {
                        weight.data[i] = LowerLimit + Math.random() * (UpperLimit - LowerLimit);
                    }
                }
            }
        } else {
            System.out.println("Верхняя граница больше нижней заданной!");
        }
    }

    public void Conv2DCreate(int KernelSize, int NumOfImagesNextLayer, int Stride, IActivation Activation, double dropout) {
        if (NumberOfLayers == 0) {
            int ImageSize = InputForm[0].length / Stride;
            if(InputForm[0].length % Stride != 0) {
                ImageSize++;
            }
            Layers.add(new Layer(ImageSize, NumOfImagesNextLayer, Activation));
            Weights.add(new Weight(KernelSize, InputForm.length, NumOfImagesNextLayer, Stride, dropout));
        } else {
            int ImageSize = Layers.get(NumberOfLayers - 1).sizeData[1] / Stride;
            if(Layers.get(NumberOfLayers - 1).sizeData[1] % Stride != 0) {
                ImageSize++;
            }
            Layers.add(new Layer(ImageSize, NumOfImagesNextLayer, Activation));
            Weights.add(new Weight(KernelSize, Layers.get(NumberOfLayers - 1).sizeData[0], NumOfImagesNextLayer, Stride, dropout));
        }
        NumberOfLayers++;
    }

    public void DenseCreate(int NumberOfNeurons, IActivation Activation, double dropout ) {
        if (NumberOfLayers == 0) {
            int imageSize = InputForm.length * InputForm[0].length * InputForm[0][0].length;
            Layers.add(new Layer(NumberOfNeurons, Activation));
            Weights.add(new Weight(imageSize, NumberOfNeurons, dropout));
        }
        else {
            Layers.add(new Layer(NumberOfNeurons, Activation));
            Weights.add(new Weight(Layers.get(NumberOfLayers - 1).data.length, NumberOfNeurons, dropout));
        }
        NumberOfLayers++;
    }

    public void MaxPoolingCreate(int Stride) {
        Layers.add(new Layer(Stride, Layers.get(NumberOfLayers - 1).sizeData));
        Weights.add(new Weight());
        NumberOfLayers++;
    }

    public void FillIndexDropout() {
        for(Weight weight: Weights) {
            if (!weight.type.equals(TypeOfLayer.MaxPooling)) {
                for (int i = 0; i < weight.dropoutIndex.length; i++) {
                    boolean flag = true;
                    while (flag) {
                        weight.dropoutIndex[i] = (int) (Math.random() * (weight.data.length - 1));
                        int count = 0;
                        for (int j = 0; j < i; j++) {
                            if (weight.dropoutIndex[j] != weight.dropoutIndex[i]) {
                                count++;
                            }
                        }
                        flag = count == i - 1;
                    }
                }
            }
        }
    }

    public void PrintIndexDropout() {
        for(Weight weight: Weights) {
            for (int i = 0; i < weight.dropoutIndex.length; i++) {
                System.out.print(weight.dropoutIndex[i] + "; ");
            }
            System.out.println();
            System.out.println("_____________________________________________________");
            System.out.println("Всего: " + weight.dropoutIndex.length);
        }
    }

    public Model clone() throws CloneNotSupportedException{
        return (Model) super.clone();
    }
}
