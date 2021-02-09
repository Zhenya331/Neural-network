package Model;

public class Weight implements Cloneable{
    public double[] data;
    public String type;
    public int[] sizeData;
    public int stride;
    public double dropout;
    public int[] dropoutIndex;

    // Convolution layer
    public Weight(int KernelSize, int NumOfImagesPrevLayer, int NumOfImagesNextLayer, int Stride, double Dropout) {
        data = new double[KernelSize * KernelSize * NumOfImagesPrevLayer * NumOfImagesNextLayer];
        type = TypeOfLayer.Conv2D;
        sizeData = new int[] {NumOfImagesNextLayer, NumOfImagesPrevLayer, KernelSize, KernelSize};
        stride = Stride;
        dropout = Dropout;
        dropoutIndex = new int[(int) (dropout * data.length)];
        //System.out.println("Conv2D weights: " + sizeData[0] + "*" + sizeData[1] + "*" + sizeData[2] + "*" + sizeData[3]);
    }

    // Dense layer
    public Weight(int NumOfNeuronsPrevLayer, int NumOfNeuronsNextLayer, double Dropout) {
        data = new double[NumOfNeuronsPrevLayer * NumOfNeuronsNextLayer];
        type = TypeOfLayer.Dense;
        sizeData = new int[] {NumOfNeuronsNextLayer, NumOfNeuronsPrevLayer};
        dropout = Dropout;
        dropoutIndex = new int[(int) (dropout * data.length)];
        //System.out.println("Dense weights: " + sizeData[0] + "*" + sizeData[1]);
    }

    // MaxPooling layer
    public Weight() {
        type = TypeOfLayer.MaxPooling;
    }

    public double[][][][] ParseDataConv2D() {
        if (type.equals(TypeOfLayer.Conv2D)) {
            double[][][][] result = new double[sizeData[0]][sizeData[1]][sizeData[2]][sizeData[3]];
            int count = 0;
            for (int i = 0; i < sizeData[0]; i++) {
                for (int j = 0; j < sizeData[1]; j++) {
                    for (int k = 0; k < sizeData[2]; k++) {
                        for (int l = 0; l < sizeData[3]; l++) {
                            result[i][j][k][l] = data[count];
                            count++;
                        }
                    }
                }
            }
            return result;
        }
        System.out.println("Неправильный тип уровня");
        return null;
    }

    public double[][] ParseDataDense() {
        if (type.equals(TypeOfLayer.Dense)) {
            double[][] result = new double[sizeData[0]][sizeData[1]];
            int count = 0;
            for (int i = 0; i < sizeData[0]; i++) {
                for (int j = 0; j < sizeData[1]; j++) {
                    result[i][j] = data[count];
                    count++;
                }
            }
            return result;
        }
        System.out.println("Неправильный тип уровня");
        return null;
    }

    public void FillDataConv2D(double[][][][] Data) {
        if (type.equals(TypeOfLayer.Conv2D)) {
            int count = 0;
            for (int i = 0; i < sizeData[0]; i++) {
                for (int j = 0; j < sizeData[1]; j++) {
                    for (int k = 0; k < sizeData[2]; k++) {
                        for (int l = 0; l < sizeData[3]; l++) {
                            data[count] = Data[i][j][k][l];
                            count++;
                        }
                    }
                }
            }
        }
        else {
            System.out.println("Неправильный тип уровня");
        }
    }

    public void FillDataDense(double[][] Data) {
        if (type.equals(TypeOfLayer.Dense)) {
            int count = 0;
            for (int i = 0; i < sizeData[0]; i++) {
                for (int j = 0; j < sizeData[1]; j++) {
                    data[count] = Data[i][j];
                    count++;
                }
            }
        }
        else {
            System.out.println("Неправильный тип уровня");
        }
    }

    public int[][] ParseIndexDropoutConv2D() {
        int[][] result = new int[dropoutIndex.length][4];
        if (type.equals(TypeOfLayer.Conv2D)) {
            int count = 0;
            for (int i = 0; i < sizeData[0]; i++) {
                for (int j = 0; j < sizeData[1]; j++) {
                    for (int k = 0; k < sizeData[2]; k++) {
                        for (int l = 0; l < sizeData[3]; l++) {
                            for (int m = 0; m < dropoutIndex.length; m++) {
                                if (count == dropoutIndex[m]) {
                                    result[m][0] = i;
                                    result[m][1] = j;
                                    result[m][2] = k;
                                    result[m][3] = l;
                                }
                            }
                            count++;
                        }
                    }
                }
            }
        }
        else {
            System.out.println("Неправильный тип уровня");
        }
        return result;
    }

    public int[][] ParseIndexDropoutDense() {
        int[][] result = new int[dropoutIndex.length][2];
        if (type.equals(TypeOfLayer.Dense)) {
            int count = 0;
            for (int i = 0; i < sizeData[0]; i++) {
                for (int j = 0; j < sizeData[1]; j++) {
                    for (int m = 0; m < dropoutIndex.length; m++) {
                        if (count == dropoutIndex[m]) {
                            result[m][0] = i;
                            result[m][1] = j;
                        }
                    }
                    count++;
                }
            }
        }
        else {
            System.out.println("Неправильный тип уровня");
        }
        return result;
    }

    public Weight clone() throws CloneNotSupportedException{
        return (Weight) super.clone();
    }
}
