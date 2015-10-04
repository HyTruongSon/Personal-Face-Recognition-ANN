// Software: Testing Neural Network ORL face database
// Author: Hy Truong Son
// Major: BSc. Computer Science
// Class: 2013 - 2016
// Institution: Eotvos Lorand University
// Email: sonpascal93@gmail.com
// Website: http://people.inf.elte.hu/hytruongson/
// Final update: October 4th, 2015
// Copyright 2015 (c) Hy Truong Son. All rights reserved. Only use for academic purposes.

import java.io.*;
import java.util.Random;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;

import MyLib.*;

public class testing {
    
    // Constants
	static String weightsFileName   = "weights.dat";
	static String samplesFileName   = "all-samples.dat";
	
	// Number of people
	static int nPeople          = 40;
	static int nPersonalSamples = 10;
	static int nSamples         = nPeople * nPersonalSamples; // = 400
	static int nTraining        = 320;
	static int nTesting         = nSamples - nTraining;

    // Original image size
	static int width        = 92;
	static int height       = 112;  
	
	// Resize to this size
	static int NewWidth     = 40;
	static int NewHeight    = 50;
    
    // Neural network architecture
	static int nInput   = NewWidth * NewHeight + 1;       // Input Layer, +1 as bias
	static int nHidden  = 128;                            // Hidden Layer
	static int nOutput  = nPeople;                        // Output Layer
	
	static double input[]   = new double [nInput];
	static double output[]  = new double [nOutput];
	
	static int scale[][]    = new int [width][height];
	static int NewScale[][] = new int [NewWidth][NewHeight];
	
	private static void Feature(String fileName) throws IOException {
		PGM256.GetGrayScale(fileName, scale, width, height);
		Normalization.Resize(scale, NewScale, width, height, NewWidth, NewHeight);

		for (int x = 0; x < NewWidth; ++x)
			for (int y = 0; y < NewHeight; ++y){
				input[y + x * NewHeight] = (double)(NewScale[x][y]) / (double)(255);
			}
	    input[NewWidth * NewHeight] = 1.0;
	}
	
	private static int testing() throws IOException {
	    System.out.println();
	    System.out.println("Testing the trained Neural Network");
	    
	    // Network initialization
	    NeuralNetwork ANN = new NeuralNetwork(nInput, nHidden, nOutput);
	    
	    // Load the weights
	    ANN.SetCost(weightsFileName);
	    
	    // Testing
	    TextFileReader file = new TextFileReader();
		file.open(samplesFileName);
		        
		int nCorrect = 0;
		
		file.ReadLine();		
		for (int sample = nTraining + 1; sample <= nSamples; ++sample) {
		    int subject = Integer.parseInt(file.ReadLine());
		    String imageFileName = file.ReadLine();
		        
		    System.out.print("Sample " + Integer.toString(sample) + ": ");
		        
		    // Feature extraction
		    Feature(imageFileName);
		        
		    // Classification
			ANN.Classification(input, output);
			    
			int predict = 0;
			for (int i = 1; i < nOutput; ++i) {
			    if (output[i] > output[predict]) {
			        predict = i;
			    }
			}
			++predict;
			    
		    if (subject == predict) {
			    ++nCorrect;
			    System.out.println("YES");
			} else {
			    System.out.println("NO");
			    System.out.println(" - Correct: " + subject);
			    System.out.println(" - Predict: " + predict);
			}
	    }
	    
	    return nCorrect;
	}
	
	public static void main(String args[]) throws IOException {
	    int nCorrect = testing();
		double percent = (double)(nCorrect) / nTesting * 100.0;
		
        System.out.println();
        System.out.println("Number of correct samples: " + nCorrect + " / " + nTesting);
        System.out.println("Accuracy: " + percent);
	}

}
