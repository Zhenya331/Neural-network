package Model;

import com.company.Main;

import java.util.ArrayList;
import java.util.Arrays;

public class Optimizer {

    private static double[][][] Conv2DTrain(double[][][] data, int[] sizeData, double[][][][] weights, int[] sizeWeights, int stride) {
        int[] sizeResult = new int[sizeData.length];
        sizeResult[0] = sizeWeights[1];
        sizeResult[1] = sizeData[1] * stride;
        sizeResult[2] = sizeData[2] * stride;
        double[][][][] weights1 = new double[sizeWeights[0]][sizeWeights[1]][sizeWeights[2]][sizeWeights[3]];
        for (int i = 0; i < sizeWeights[0]; i++) {
            for (int j = 0; j < sizeWeights[1]; j++) {
                for (int k = 0; k < sizeWeights[2]; k++) {
                    if (sizeWeights[3] >= 0)
                        System.arraycopy(weights[i][j][k], 0, weights1[i][j][k], 0, sizeWeights[3]);
                }
            }
        }
        double[][][] inputData = new double[sizeWeights[0]][sizeResult[1]][sizeResult[2]];
        for (int i = 0; i < sizeWeights[0]; i++) {
            for (int j = 0; j < sizeResult[1]; j++) {
                for (int k = 0; k < sizeResult[2]; k++) {
                    inputData[i][j][k] = 0;
                }
            }
            for (int j = 0; j < sizeData[1]; j++) {
                for (int k = 0; k < sizeData[2]; k++) {
                    inputData[i][j * stride][k * stride] = data[i][j][k];
                }
            }
            for (int j = 0; j < sizeWeights[1]; j++) {
                for (int k = 0; k < sizeWeights[2] / 2; k++) {
                    for (int l = 0; l < sizeWeights[3]; l++) {
                        double buf = weights1[i][j][k][l];
                        weights1[i][j][k][l] = weights1[i][j][sizeWeights[2] - 1 - k][sizeWeights[3] - 1 - l];
                        weights1[i][j][sizeWeights[2] - 1 - k][sizeWeights[3] - 1 - l] = buf;
                    }
                }
                for (int k = 0; k < sizeWeights[3] / 2; k++) {
                    double buf = weights1[i][j][sizeWeights[2] / 2][k];
                    weights1[i][j][sizeWeights[2] / 2][k] = weights1[i][j][sizeWeights[2] / 2][sizeWeights[3] - 1 - k];
                    weights1[i][j][sizeWeights[2] / 2][sizeWeights[3] - 1 - k] = buf;
                }
            }
        }

        int[] sizeReshapeWeights = new int[sizeWeights.length];
        sizeReshapeWeights[0] = sizeWeights[1];
        sizeReshapeWeights[1] = sizeWeights[0];
        sizeReshapeWeights[2] = sizeWeights[2];
        sizeReshapeWeights[3] = sizeWeights[3];
        double[][][][] reshapeWeights = new double[sizeReshapeWeights[0]][sizeReshapeWeights[1]][sizeReshapeWeights[2]][sizeReshapeWeights[3]];
        for (int i = 0; i < sizeReshapeWeights[0]; i++) {
            for (int j = 0; j < sizeReshapeWeights[1]; j++) {
                for (int k = 0; k < sizeReshapeWeights[2]; k++) {
                    if (sizeReshapeWeights[3] >= 0)
                        System.arraycopy(weights1[j][i][k], 0, reshapeWeights[i][j][k], 0, sizeReshapeWeights[3]);
                }
            }
        }
        return Conv2D(inputData, reshapeWeights, sizeResult, sizeReshapeWeights);
    }

    private static double[][][] Conv2D(double[][][] InputData, double[][][][] Weights, int[] sizeData, int[] sizeWeights) {
        double[][][] result = new double[sizeData[0]][sizeData[1]][sizeData[2]];
        for (int i = 0; i < sizeWeights[0]; i++) {
            for (int j = 0; j < InputData[0].length; j++) {
                for (int k = 0; k < InputData[0][0].length; k++) {
                    double res = 0;
                    for (int l = 0; l < InputData.length; l++) {
                        for (int m = 0; m < sizeWeights[2]; m++) {
                            for (int n = 0; n < sizeWeights[3]; n++) {
                                if ((j - sizeWeights[2] / 2 + m >= 0) && (j - sizeWeights[2] / 2 + m < InputData[0].length) &&
                                        (k - sizeWeights[3] / 2 + n >= 0) && (k - sizeWeights[3] / 2 + n < InputData[0][0].length)) {
                                    res += InputData[l][j - sizeWeights[2] / 2 + m][k - sizeWeights[3] / 2 + n] * Weights[i][l][m][n];
                                }
                            }
                        }
                    }
                    result[i][j][k] = res;
                }
            }
        }
        return result;
    }

    public static ArrayList<Weight> BackPropagation(ForOptimizer arg) {
        ArrayList<Weight> result = arg.Training[0].Weights;
        double alpha = arg.learningRate;
        double Error = 0;

        for (int i = 0; i < arg.BatchSize; i++) {
            double[] data = arg.Training[i].Layers.get(arg.Training[i].NumberOfLayers - 1).activation.Method(arg.Training[i].Layers.get(arg.Training[i].NumberOfLayers - 1).data);
            Error += arg.Loss.apply(new ForLoss(data, arg.Ytrain.get(i)));
        }
        Error /= arg.BatchSize;
        System.out.println("Error: " + Error);
        Log.trainError.add(Error);

        Model[] delta = new Model[arg.BatchSize];
        for (int i = 0; i < arg.BatchSize; i++) {
            delta[i] = new Model();
            delta[i].NumberOfLayers = arg.Training[i].NumberOfLayers;
            delta[i].InputForm = new double[arg.Training[i].InputForm.length][arg.Training[i].InputForm[0].length][arg.Training[i].InputForm[0][0].length];
            delta[i].Weights = new ArrayList<>();
            delta[i].Layers = new ArrayList<>();
            int count = 0;
            for (Weight weight: arg.Training[i].Weights) {
                if (!weight.type.equals(TypeOfLayer.MaxPooling)) {
                    try {
                        delta[i].Weights.add(weight.clone());
                    } catch (CloneNotSupportedException e) {
                        e.printStackTrace();
                    }
                    delta[i].Weights.get(count).data = new double[arg.Training[i].Weights.get(count).data.length];
                    System.arraycopy(arg.Training[i].Weights.get(count).data, 0, delta[i].Weights.get(count).data, 0, arg.Training[i].Weights.get(count).data.length);
                }
                else {
                    delta[i].Weights.add(new Weight());
                }
                count++;
            }
            count = 0;
            for (Layer layer: arg.Training[i].Layers) {
                try {
                    delta[i].Layers.add(layer.clone());
                } catch (CloneNotSupportedException e) {
                    e.printStackTrace();
                }
                delta[i].Layers.get(count).data = new double[arg.Training[i].Layers.get(count).data.length];
                System.arraycopy(arg.Training[i].Layers.get(count).data, 0, delta[i].Layers.get(count).data, 0, arg.Training[i].Layers.get(count).data.length);
                count++;
            }
            for (int k = 0; k < arg.Training[i].InputForm.length; k++) {
                for (int l = 0; l < arg.Training[i].InputForm[0].length; l++) {
                    System.arraycopy(arg.Training[i].InputForm[k][l], 0, delta[i].InputForm[k][l], 0, arg.Training[i].InputForm[0][0].length);
                }
            }
        }
        for (int i = arg.Training[0].NumberOfLayers - 1; i >= 0; i--) {
            if (i == arg.Training[0].NumberOfLayers - 1) {
                for (int j = 0; j < arg.BatchSize; j++) {
                    double[] resDerivative = arg.Training[j].Layers.get(i).activation.Derivative(arg.Training[j].Layers.get(i).data);
                    double[] resActivation = arg.Training[j].Layers.get(i).activation.Method(arg.Training[j].Layers.get(i).data);
                    for (int k = 0; k < delta[j].Layers.get(i).sizeData[0]; k++) {
                        delta[j].Layers.get(i).data[k] = resActivation[k];
                        if (arg.Ytrain.get(j) == k) {
                            delta[j].Layers.get(i).data[k] = resActivation[k] - 1;
                        }
                        delta[j].Layers.get(i).data[k] *= resDerivative[k];
                    }
                    //System.out.println();System.out.println();System.out.println("ОБУЧЕНИЕ:"); Main.PrintMassive(delta[j].Layers.get(i).data, delta[j].Layers.get(i).sizeData);

                }
                continue;
            }

            if (delta[0].Layers.get(i).type.equals(TypeOfLayer.Dense)) {
                for (int j = 0; j < arg.BatchSize; j++) {
                    double[][] weights1 = result.get(i + 1).ParseDataDense();
                    int[] sizeWeights1 = arg.Training[j].Weights.get(i + 1).sizeData;
                    double[] inputData1 = arg.Training[j].Layers.get(i).data;
                    inputData1 = arg.Training[j].Layers.get(i).activation.Method(inputData1);
                    double[] deltas1 = delta[j].Layers.get(i + 1).data;
                    int[][] indexDropout = result.get(i + 1).ParseIndexDropoutDense();

                    for (int k = 0; k < sizeWeights1[0]; k++) {
                        for (int l = 0; l < sizeWeights1[1]; l++) {
                            boolean flag = true;
                            for (int m = 0; m < result.get(i + 1).dropoutIndex.length; m++) {
                                if (k == indexDropout[m][0] && l == indexDropout[m][1]) {
                                    flag = false;
                                    break;
                                }
                            }
                            if (flag) {
                                weights1[k][l] -= alpha * deltas1[k] * inputData1[l];// / arg.BatchSize;
                            }
                        }
                    }
                    result.get(i + 1).FillDataDense(weights1);
                    //Main.PrintMassive(result.get(i + 1).ParseDataDense(), result.get(i + 1).sizeData);
                }

                for (int j = 0; j < arg.BatchSize; j++) {
                    double[] resDerivative = arg.Training[j].Layers.get(i).activation.Derivative(arg.Training[j].Layers.get(i).data);
                    double[][] weights = result.get(i + 1).ParseDataDense();
                    for (int k = 0; k < delta[j].Layers.get(i).sizeData[0]; k++) {
                        delta[j].Layers.get(i).data[k] = 0;
                        for (int l = 0; l < delta[j].Layers.get(i + 1).sizeData[0]; l++) {
                            delta[j].Layers.get(i).data[k] += delta[j].Layers.get(i + 1).data[l] * weights[l][k];
                        }
                        delta[j].Layers.get(i).data[k] *= resDerivative[k];
                    }
                    //Main.PrintMassive(delta[j].Layers.get(i).data, delta[j].Layers.get(i).sizeData);
                }
                continue;
            }

            if (delta[0].Layers.get(i).type.equals(TypeOfLayer.Conv2D) || delta[0].Layers.get(i).type.equals(TypeOfLayer.MaxPooling)) {
                if (delta[0].Layers.get(i + 1).type.equals(TypeOfLayer.MaxPooling)) {
                    for (int j = 0; j < arg.BatchSize; j++) {
                        double[][][] deltas = delta[j].Layers.get(i + 1).ParseDataConv2D();
                        int[][][] indexMaxPooling = delta[j].Layers.get(i + 1).indexMaxPooling;
                        double[][][] resUnPooling = delta[j].Layers.get(i).ParseDataConv2D();

                        for (int k = 0; k < resUnPooling.length; k++) {
                            for (int l = 0; l < resUnPooling[0].length; l++) {
                                for (int m = 0; m < resUnPooling[0][0].length; m++) {
                                    resUnPooling[k][l][m] = 0.0;
                                }
                            }
                        }

                        for (int k = 0; k < deltas.length; k++) {
                            int count = 0;
                            for (int l = 0; l < deltas[0].length; l++) {
                                for (int m = 0; m < deltas[0][0].length; m++) {
                                    resUnPooling[k][indexMaxPooling[k][count][0]][indexMaxPooling[k][count][1]] = deltas[k][l][m];
                                    count++;
                                }
                            }
                        }
                        delta[j].Layers.get(i).FillDataConv2D(resUnPooling);
                        //Main.PrintMassive(resUnPooling, delta[j].Layers.get(i).sizeData);
                    }
                    continue;
                }

                if (delta[0].Layers.get(i + 1).type.equals(TypeOfLayer.Dense)) {
                    for (int j = 0; j < arg.BatchSize; j++) {
                        double[][] weights1 = result.get(i + 1).ParseDataDense();
                        int[] sizeWeights1 = arg.Training[j].Weights.get(i + 1).sizeData;
                        double[] inputData1 = arg.Training[j].Layers.get(i).data;
                        inputData1 = arg.Training[j].Layers.get(i).activation.Method(inputData1);
                        double[] deltas1 = delta[j].Layers.get(i + 1).data;
                        int[][] indexDropout = result.get(i + 1).ParseIndexDropoutDense();

                        for (int k = 0; k < sizeWeights1[0]; k++) {
                            for (int l = 0; l < sizeWeights1[1]; l++) {
                                boolean flag = true;
                                for (int m = 0; m < result.get(i + 1).dropoutIndex.length; m++) {
                                    if (k == indexDropout[m][0] && l == indexDropout[m][1]) {
                                        flag = false;
                                        break;
                                    }
                                }
                                if (flag) {
                                    weights1[k][l] -= alpha * deltas1[k] * inputData1[l];// / arg.BatchSize;
                                }
                            }
                        }
                        result.get(i + 1).FillDataDense(weights1);
                        //Main.PrintMassive(result.get(i + 1).ParseDataDense(), result.get(i + 1).sizeData);
                    }

                    for (int j = 0; j < arg.BatchSize; j++) {
                        double[] resDerivative = arg.Training[j].Layers.get(i).activation.Derivative(arg.Training[j].Layers.get(i).data);
                        double[][] weights = result.get(i + 1).ParseDataDense();
                        for (int k = 0; k < delta[j].Layers.get(i).data.length; k++) {
                            delta[j].Layers.get(i).data[k] = 0;
                            for (int l = 0; l < delta[j].Layers.get(i + 1).sizeData[0]; l++) {
                                delta[j].Layers.get(i).data[k] += delta[j].Layers.get(i + 1).data[l] * weights[l][k];
                            }
                            delta[j].Layers.get(i).data[k] *= resDerivative[k];
                        }
                        //Main.PrintMassive(delta[j].Layers.get(i).ParseDataConv2D(), delta[j].Layers.get(i).sizeData);
                    }
                } else {
                    for (int j = 0; j < arg.BatchSize; j++) {
                        double[][][][] weights1 = result.get(i + 1).ParseDataConv2D();
                        int[] sizeWeights1 = arg.Training[j].Weights.get(i + 1).sizeData;

                        double[] bufData1 = arg.Training[j].Layers.get(i).data;
                        arg.Training[j].Layers.get(i).data = arg.Training[j].Layers.get(i).activation.Method(bufData1);
                        double[][][] inputImages1 = arg.Training[j].Layers.get(i).ParseDataConv2D();
                        arg.Training[j].Layers.get(i).data = bufData1;

                        double[][][] deltas1 = delta[j].Layers.get(i + 1).ParseDataConv2D();
                        int[] sizeDeltas1 = delta[j].Layers.get(i + 1).sizeData;
                        int[][] indexDropout = result.get(i + 1).ParseIndexDropoutConv2D();

                        for (int k = 0; k < sizeWeights1[0]; k++) {
                            for (int l = 0; l < sizeWeights1[1]; l++) {
                                for (int m = 0; m < sizeWeights1[2]; m++) {
                                    for (int n = 0; n < sizeWeights1[3]; n++) {
                                        boolean flag = true;
                                        for (int p = 0; p < result.get(i + 1).dropoutIndex.length; p++) {
                                            if (k == indexDropout[p][0] && l == indexDropout[p][1] && m == indexDropout[p][2] && m == indexDropout[p][3]) {
                                                flag = false;
                                                break;
                                            }
                                        }
                                        if (flag) {
                                            for (int p = 0; p < sizeDeltas1[1]; p++) {
                                                for (int s = 0; s < sizeDeltas1[2]; s++) {
                                                    int[] index = new int[2];
                                                    index[0] = p * delta[j].Weights.get(i + 1).stride;
                                                    index[1] = s * delta[j].Weights.get(i + 1).stride;
                                                    int[] indexWeight = new int[2];
                                                    indexWeight[0] = index[0] + m - sizeWeights1[2] / 2;
                                                    indexWeight[1] = index[1] + n - sizeWeights1[3] / 2;
                                                    if (indexWeight[0] >= 0 && indexWeight[0] < inputImages1[0].length && indexWeight[1] >= 0 && indexWeight[1] < inputImages1[0][0].length) {
                                                        weights1[k][l][m][n] -= alpha * deltas1[k][p][s] * inputImages1[l][indexWeight[0]][indexWeight[1]];// / arg.BatchSize;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        result.get(i + 1).FillDataConv2D(weights1);
                        //Main.PrintMassive(result.get(i + 1).ParseDataConv2D(), result.get(i + 1).sizeData);
                    }

                    for (int j = 0; j < arg.BatchSize; j++) {
                        double[] resDerivative = arg.Training[j].Layers.get(i).activation.Derivative(arg.Training[j].Layers.get(i).data);
                        double[][][][] weights = result.get(i + 1).ParseDataConv2D();
                        int[] sizeDataWeights = arg.Training[j].Weights.get(i + 1).sizeData;
                        double[][][] deltaData = delta[j].Layers.get(i + 1).ParseDataConv2D();
                        int stride = arg.Training[j].Weights.get(i + 1).stride;

                        double[][][] dataLayer = Conv2DTrain(deltaData, delta[j].Layers.get(i + 1).sizeData, weights, sizeDataWeights, stride);

                        delta[j].Layers.get(i).FillDataConv2D(dataLayer);

                        for (int k = 0; k < delta[j].Layers.get(i).data.length; k++) {
                            delta[j].Layers.get(i).data[k] *= resDerivative[k];
                        }
                        //Main.PrintMassive(delta[j].Layers.get(i).ParseDataConv2D(), delta[j].Layers.get(i).sizeData);
                    }
                }
            }
        }

        if (result.get(0).type.equals(TypeOfLayer.Conv2D)) {
            for (int j = 0; j < arg.BatchSize; j++) {
                double[][][][] weights1 = result.get(0).ParseDataConv2D();
                int[] sizeWeights1 = arg.Training[j].Weights.get(0).sizeData;
                double[][][] inputImages1 = arg.Training[j].InputForm;
                double[][][] deltas1 = delta[j].Layers.get(0).ParseDataConv2D();
                int[] sizeDeltas1 = delta[j].Layers.get(0).sizeData;

                int[][] indexDropout = result.get(0).ParseIndexDropoutConv2D();

                for (int k = 0; k < sizeWeights1[0]; k++) {
                    for (int l = 0; l < sizeWeights1[1]; l++) {
                        for (int m = 0; m < sizeWeights1[2]; m++) {
                            for (int n = 0; n < sizeWeights1[3]; n++) {
                                boolean flag = true;
                                for (int p = 0; p < result.get(0).dropoutIndex.length; p++) {
                                    if (k == indexDropout[p][0] && l == indexDropout[p][1] && m == indexDropout[p][2] && m == indexDropout[p][3]) {
                                        flag = false;
                                        break;
                                    }
                                }
                                if (flag) {
                                    for (int p = 0; p < sizeDeltas1[1]; p++) {
                                        for (int s = 0; s < sizeDeltas1[2]; s++) {
                                            int[] index = new int[2];
                                            index[0] = p * delta[j].Weights.get(0).stride;
                                            index[1] = s * delta[j].Weights.get(0).stride;
                                            int[] indexWeight = new int[2];
                                            indexWeight[0] = index[0] + m - sizeWeights1[2] / 2;
                                            indexWeight[1] = index[1] + n - sizeWeights1[3] / 2;
                                            if (indexWeight[0] >= 0 && indexWeight[0] < inputImages1[0].length && indexWeight[1] >= 0 && indexWeight[1] < inputImages1[0][0].length) {
                                                weights1[k][l][m][n] -= alpha * deltas1[k][p][s] * inputImages1[l][indexWeight[0]][indexWeight[1]];// / arg.BatchSize;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                result.get(0).FillDataConv2D(weights1);
                //Main.PrintMassive(result.get(0).ParseDataConv2D(), result.get(0).sizeData); System.out.println("КОНЕЦ ОБУЧЕНИЯ....");
            }
        }
        if (result.get(0).type.equals(TypeOfLayer.Dense)) {
            for (int j = 0; j < arg.BatchSize; j++) {
                double[][] weights1 = result.get(0).ParseDataDense();
                int[] sizeWeights1 = arg.Training[j].Weights.get(0).sizeData;
                double[][][] InputImage = arg.Training[j].InputForm;
                double[] inputData1 = new double[InputImage.length * InputImage[0].length * InputImage[0][0].length];
                int count = 0;
                for (double[][] doubles : InputImage) {
                    for (int k = 0; k < InputImage[0].length; k++) {
                        for (int l = 0; l < InputImage[0][0].length; l++) {
                            inputData1[count] = doubles[k][l];
                            count++;
                        }
                    }
                }
                double[] deltas1 = delta[j].Layers.get(0).data;
                int[][] indexDropout = result.get(0).ParseIndexDropoutDense();

                for (int k = 0; k < sizeWeights1[0]; k++) {
                    for (int l = 0; l < sizeWeights1[1]; l++) {
                        boolean flag = true;
                        for (int m = 0; m < result.get(0).dropoutIndex.length; m++) {
                            if (k == indexDropout[m][0] && l == indexDropout[m][1]) {
                                flag = false;
                                break;
                            }
                        }
                        if (flag) {
                            weights1[k][l] -= alpha * deltas1[k] * inputData1[l];// / arg.BatchSize;
                        }
                    }
                }
                result.get(0).FillDataDense(weights1);
                //Main.PrintMassive(result.get(0).ParseDataDense(), result.get(0).sizeData); System.out.println("КОНЕЦ ОБУЧЕНИЯ....");
            }
        }
        return result;
    }
}