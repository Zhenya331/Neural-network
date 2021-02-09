package Activation;

public class SoftMax implements IActivation{

    @Override
    public double[] Method(double[] data) {
        double[] result = new double[data.length];
        double sum = 0;
        for (double datum : data) {
            sum += Math.exp(datum);
        }
        for (int i = 0; i < data.length; i++) {
            result[i] = Math.exp(data[i]) / sum;
        }
        return result;
    }

    @Override
    public double[] Derivative(double[] data) {
        double[] result = Method(data);
        for (int i = 0; i < data.length; i++) {
            result[i] = result[i] * (1 - result[i]);
        }
        return result;
    }
}
