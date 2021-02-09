package Activation;

public class ReLu implements IActivation{

    @Override
    public double[] Method(double[] data) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            if(data[i] < 0) {
                result[i] = 0;
            }
            else {
                result[i] = data[i];
            }
        }
        return result;
    }

    @Override
    public double[] Derivative(double[] data) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            if(data[i] < 0) {
                result[i] = 0;
            }
            else {
                result[i] = 1;
            }
        }
        return result;
    }
}
