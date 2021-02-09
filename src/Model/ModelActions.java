package Model;

import com.company.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.Function;

public class ModelActions {
    public Model model;

    public ModelActions(Model model) {
        this.model = model;
    }

    private double[][][] Conv2D(double[][][] InputData, double[][][][] Weights, int[] sizeData, int[] sizeWeights, int Stride) {
        double[][][] result = new double[sizeData[0]][sizeData[1]][sizeData[2]];
        for(int i = 0; i < sizeWeights[0]; i++) {
            for(int j = 0; j < InputData[0].length; j += Stride) {
                for(int k = 0; k < InputData[0][0].length; k += Stride) {
                    double res = 0;
                    for(int l = 0; l < InputData.length; l++) {
                        for(int m = 0; m < sizeWeights[2]; m++) {
                            for(int n = 0; n < sizeWeights[3]; n++) {
                                if((j - sizeWeights[2] / 2 + m >= 0) && (j - sizeWeights[2] / 2 + m < InputData[0].length) &&
                                        (k - sizeWeights[3] / 2 + n >= 0) && (k - sizeWeights[3] / 2 + n < InputData[0][0].length)) {
                                    res += InputData[l][j - sizeWeights[2] / 2 + m][k - sizeWeights[3] / 2 + n] * Weights[i][l][m][n];
                                }
                            }
                        }
                    }
                    result[i][j / Stride][k / Stride] = res;
                }
            }
        }
        //System.out.println("Размер уровня (Conv2D):" + result.length + "*" + result[0].length + "*" + result[0][0].length);
        return result;
    }

    private double[] Dense(double[] InputData, double[][] Weights, int[] sizeWeights) {
        double[] result = new double[sizeWeights[0]];
        for(int i = 0; i < sizeWeights[0]; i++) {
            double res = 0;
            for(int j = 0; j < sizeWeights[1]; j++) {
                res += InputData[j] * Weights[i][j];
            }
            result[i] = res;
        }
        //System.out.println("Размер уровня (Dense):" + result.length);
        return result;
    }

    private double[][][] MaxPooling(double[][][] inputData, int[][][] indexResult, int stride) {
        double[][][] result = new double[inputData.length][inputData[0].length / stride][inputData[0][0].length / stride];
        for (int i = 0; i < inputData.length; i++) {
            int count = 0;
            for (int j = 0; j < inputData[0].length; j += stride) {
                for (int k = 0; k < inputData[0][0].length; k += stride) {
                    double max = 0;
                    for (int l = j; l < j + stride; l++) {
                        for (int m = k; m < k + stride; m++) {
                            if (inputData[i][l][m] > max) {
                                max = inputData[i][l][m];
                                indexResult[i][count][0] = l;
                                indexResult[i][count][1] = m;
                            }
                        }
                    }
                    count++;
                    result[i][j / stride][k / stride] = max;
                }
            }
        }
        return result;
    }

    private double[] Predict(double[][][] InputImage) {
        double[] result;
        double[] buf;
        for(int i = 0; i < model.NumberOfLayers; i++) {
            if(i == 0) {
                if (model.Weights.get(i).type.equals(TypeOfLayer.Conv2D)) {
                    model.InputForm = InputImage;
                    double[][][] res = Conv2D(InputImage, model.Weights.get(i).ParseDataConv2D(), model.Layers.get(i).sizeData, model.Weights.get(i).sizeData, model.Weights.get(i).stride);
                    model.Layers.get(i).FillDataConv2D(res);
                    //Main.PrintMassive(InputImage, new int[]{InputImage.length, InputImage[0].length, InputImage[0][0].length});
                    //Main.PrintMassive(model.Weights.get(i).ParseDataConv2D(), model.Weights.get(i).sizeData);
                    //Main.PrintMassive(model.Layers.get(i).ParseDataConv2D(), model.Layers.get(i).sizeData);
                }
                if (model.Weights.get(i).type.equals(TypeOfLayer.Dense)) {
                    model.InputForm = InputImage;
                    double[] inputData = new double[InputImage.length * InputImage[0].length * InputImage[0][0].length];
                    int count = 0;
                    for (double[][] doubles : InputImage) {
                        for (int k = 0; k < InputImage[0].length; k++) {
                            for (int l = 0; l < InputImage[0][0].length; l++) {
                                inputData[count] = doubles[k][l];
                                count++;
                            }
                        }
                    }
                    double[] res = Dense(inputData, model.Weights.get(i).ParseDataDense(), model.Weights.get(i).sizeData);
                    model.Layers.get(i).FillDataDense(res);
                    //Main.PrintMassive(InputImage, new int[]{InputImage.length, InputImage[0].length, InputImage[0][0].length});
                    //Main.PrintMassive(model.Weights.get(i).ParseDataDense(), model.Weights.get(i).sizeData);
                    //Main.PrintMassive(model.Layers.get(i).data, model.Layers.get(i).sizeData);
                }
                continue;
            }

            if (model.Layers.get(i).type.equals(TypeOfLayer.MaxPooling)) {
                double[][][] res = MaxPooling(model.Layers.get(i - 1).ParseDataConv2D(), model.Layers.get(i).indexMaxPooling, model.Layers.get(i).stride);
                model.Layers.get(i).FillDataConv2D(res);
                //Main.PrintMassive(res, model.Layers.get(i).sizeData);
                continue;
            }

            buf = model.Layers.get(i - 1).data;

            if(model.Weights.get(i).type.equals(TypeOfLayer.Conv2D)) {
                model.Layers.get(i - 1).data = model.Layers.get(i - 1).activation.Method(model.Layers.get(i - 1).data);
                double[][][] res = Conv2D(model.Layers.get(i - 1).ParseDataConv2D(), model.Weights.get(i).ParseDataConv2D(), model.Layers.get(i).sizeData, model.Weights.get(i).sizeData, model.Weights.get(i).stride);
                model.Layers.get(i - 1).data = buf;
                model.Layers.get(i).FillDataConv2D(res);
                //Main.PrintMassive(model.Weights.get(i).ParseDataConv2D(), model.Weights.get(i).sizeData);
                //Main.PrintMassive(model.Layers.get(i).ParseDataConv2D(), model.Layers.get(i).sizeData);
                continue;
            }
            if(model.Weights.get(i).type.equals(TypeOfLayer.Dense)) {
                model.Layers.get(i - 1).data = model.Layers.get(i - 1).activation.Method(model.Layers.get(i - 1).data);
                double[] res = Dense(model.Layers.get(i - 1).data, model.Weights.get(i).ParseDataDense(), model.Weights.get(i).sizeData);
                model.Layers.get(i - 1).data = buf;
                model.Layers.get(i).FillDataDense(res);
                //Main.PrintMassive(model.Weights.get(i).ParseDataDense(), model.Weights.get(i).sizeData);
                //Main.PrintMassive(model.Layers.get(i).data, model.Layers.get(i).sizeData);
                if (i == model.NumberOfLayers - 1) {
                    //Main.PrintMassive(model.Weights.get(i).ParseDataDense(), model.Weights.get(i).sizeData);
                    //Main.PrintMassive(model.Layers.get(i).data, model.Layers.get(i).sizeData);
                }
            }
        }
        result = model.Layers.get(model.NumberOfLayers - 1).activation.Method(model.Layers.get(model.NumberOfLayers - 1).data);
        return result;
    }

    public void Fit(ForDataset dataTrain, ForDataset dataTest, int BatchSize, int Epochs, Function<ForLoss, Double> Loss, Function<ForOptimizer, ArrayList<Weight>> Optimizer, double learningRate) throws CloneNotSupportedException {
        Model[] Training = new Model[BatchSize];
        for(int epoch = 0; epoch < Epochs; epoch++) {
            System.out.println("Epoch: " + epoch);
            FileWorkMNIST.Shuffle(dataTrain);
            for (int i = 0; i < dataTrain.num; i += BatchSize) {
                if (i + BatchSize > dataTrain.num) { break; }
                ArrayList<Integer> answerBatch = new ArrayList<>();
                model.FillIndexDropout();
                for (int j = 0; j < BatchSize; j++) {
                    //System.out.println(i + j + ") " + dataTrain.answers.get(i + j));
                    //System.out.println(data.fileNames.get(i + j));
                    double[][][] inputImage = ImageWork.ReshapeImage(ImageWork.convertImageToPixelsMNIST(FileWorkMNIST.pathToDataset + dataTrain.fileNames.get(i + j)));
                    double[][][] inputImagePool = ImageWork.PoolImage(inputImage, 1);
                    answerBatch.add(dataTrain.answers.get(i + j));
                    double[] result = Predict(inputImagePool);
                    //System.out.println(Arrays.toString(model.Layers.get(model.NumberOfLayers - 1).data));
                    Training[j] = new Model();
                    Training[j].NumberOfLayers = model.NumberOfLayers;
                    Training[j].InputForm = new double[model.InputForm.length][model.InputForm[0].length][model.InputForm[0][0].length];
                    Training[j].Weights = new ArrayList<>();
                    Training[j].Layers = new ArrayList<>();
                    int count = 0;
                    for (Weight weight: model.Weights) {
                        if (!weight.type.equals(TypeOfLayer.MaxPooling)) {
                            Training[j].Weights.add(weight.clone());
                            Training[j].Weights.get(count).data = new double[model.Weights.get(count).data.length];
                            System.arraycopy(model.Weights.get(count).data, 0, Training[j].Weights.get(count).data, 0, model.Weights.get(count).data.length);
                        }
                        else {
                            Training[j].Weights.add(new Weight());
                        }
                        count++;
                    }
                    count = 0;
                    for (Layer layer: model.Layers) {
                        Training[j].Layers.add(layer.clone());
                        Training[j].Layers.get(count).data = new double[model.Layers.get(count).data.length];
                        System.arraycopy(model.Layers.get(count).data, 0, Training[j].Layers.get(count).data, 0, model.Layers.get(count).data.length);
                        count++;
                    }
                    for (int k = 0; k < model.InputForm.length; k++) {
                        for (int l = 0; l < model.InputForm[0].length; l++) {
                            System.arraycopy(model.InputForm[k][l], 0, Training[j].InputForm[k][l], 0, model.InputForm[0][0].length);
                        }
                    }
                    //for (double v : result) { System.out.print(v + "; ");} System.out.println();
                }
                for (int j = 0; j < BatchSize; j++) {
                    //Main.PrintMassive(Training[j].InputForm, new int[] {Training[j].InputForm.length, Training[j].InputForm[0].length, Training[j].InputForm[0][0].length});
                    //System.out.println(Arrays.toString(Training[j].Layers.get(model.NumberOfLayers - 1).data));
                }
                int progress = i + BatchSize;
                System.out.print(epoch + "): " + "(" + progress + "/" + dataTrain.num + "): ");
                model.Weights = Optimizer.apply(new ForOptimizer(Training, answerBatch, BatchSize, Loss, learningRate));
            }
            Log.GetTrainError(epoch);
            TrainAccuracy(dataTrain);
            Evaluate(dataTest);
        }
        Log.GetTestAccuracy();
        Log.GetTrainAccuracy();
    }

    public void Evaluate(ForDataset data) {
        System.out.println();
        System.out.println();
        System.out.println("Test:");
        int count = 0;
        for (int i = 0; i < data.num; i++) {
            double[][][] inputImage = ImageWork.ReshapeImage(ImageWork.convertImageToPixelsMNIST(FileWorkMNIST.pathToDataset + data.fileNames.get(i)));
            double[] result = Predict(inputImage);
            double max = result[0];
            int index = 0;
            for (int j = 0; j < result.length; j++) {
                if (result[j] > max) {
                    max = result[j];
                    index = j;
                }
            }
            if (index == data.answers.get(i)) {
                count++;
            }
        }
        System.out.println("Правильно опознано " + count + " из " + data.num);
        double accuracy = (double) count / data.num;
        Log.testAccuracy.add(accuracy);
        System.out.println();
        System.out.println();
    }

    public void TrainAccuracy(ForDataset data) {
        int count = 0;
        for (int i = 0; i < data.num; i++) {
            double[][][] inputImage = ImageWork.ReshapeImage(ImageWork.convertImageToPixelsMNIST(FileWorkMNIST.pathToDataset + data.fileNames.get(i)));
            double[] result = Predict(inputImage);
            double max = result[0];
            int index = 0;
            for (int j = 0; j < result.length; j++) {
                if (result[j] > max) {
                    max = result[j];
                    index = j;
                }
            }
            if (index == data.answers.get(i)) {
                count++;
            }
        }
        double accuracy = (double) count / data.num;
        Log.trainAccuracy.add(accuracy);
    }
}
