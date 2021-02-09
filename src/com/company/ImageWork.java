package com.company;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class ImageWork {

    public static double[][][] ReshapeImage(int[][][] imageData) {
        double[][][] result = new double[imageData[0][0].length][imageData.length][imageData[0].length];
        for (int i = 0; i < imageData[0][0].length; i++) {
            for (int j = 0; j < imageData.length; j++) {
                for (int k = 0; k < imageData.length; k++) {
                    result[i][j][k] = imageData[j][k][i];
                    result[i][j][k] /= 255.0;
                }
            }
        }
        return result;
    }

    public static double[][][] PoolImage(double[][][] imageData, int stride) {
        double[][][] result = new double[imageData.length][imageData[0].length / stride][imageData[0][0].length / stride];
        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[0].length; j++) {
                for (int k = 0; k < result[0][0].length; k++) {
                    //double bufresult = 0;
                    double bufresult = imageData[i][j * stride][k * stride];
                    for (int l = j * stride; l < j * stride + stride; l++) {
                        for (int m = k * stride; m < k * stride + stride; m++) {
                            //bufresult += imageData[i][l][m] / stride / stride;
                            if (imageData[i][l][m] > bufresult) { bufresult = imageData[i][l][m]; }
                        }
                    }
                    result[i][j][k] = bufresult;
                }
            }
        }
        return result;
    }

    public static int[][][] convertImageToPixelsMNIST(String imagePath) {
        BufferedImage image = null;
        try {
            image = ImageIO.read(new File(imagePath));
        } catch (IOException e) {
            e.printStackTrace();
        }
        assert image != null;
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();

        int[][][] result = new int[image.getWidth()][image.getHeight()][1];
        int counter = 0;
        for(int i = 0; i < image.getWidth(); i++) {
            for(int j = 0; j < image.getHeight(); j++) {
                result[i][j][0] = pixels[counter];
                if (result[i][j][0] < 0) { result[i][j][0] = 256 + result[i][j][0]; }
                counter++;
            }
        }
        return result;
    }
}
