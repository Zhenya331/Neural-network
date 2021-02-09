package Activation;

public class Sigmoid implements IActivation{
    @Override
    public double[] Method(double[] data) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = 1.0 / (1.0 + Math.exp( - data[i]));
        }
        return result;
    }

    @Override
    public double[] Derivative(double[] data) {
        double[] bufres = Method(data);
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = bufres[i] * (1.0 - bufres[i]);
        }
        return result;
    }
}
