// Software: PGM image file processing library
// Author: Hy Truong Son
// Major: BSc. Computer Science
// Class: 2013 - 2016
// Institution: Eotvos Lorand University
// Email: sonpascal93@gmail.com
// Website: http://people.inf.elte.hu/hytruongson/
// Final update: October 4th, 2015
// Copyright 2015 (c) Hy Truong Son. All rights reserved. Only use for academic purposes.

package MyLib;

import java.lang.*;
import java.io.*;
import java.awt.*;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Iterator;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;

public class PGM256 {
	
	public static int RGB(int red,int green,int blue){
		return (0xff000000)|(red<<16)|(green<<8)|blue;
	}
	
	public static String GetType(String name){
		int i,j;
		String res;
		j=0;
		for (i=0;i<name.length();i++)
			if (name.charAt(i)=='.'){
				j=i;
				break;
			}
		res="";
		for (i=j+1;i<name.length();i++) res+=name.charAt(i);
		return res;
	}
	
	public static int GetNumber(int num[],int n){
		int i,j,res;
		res=0;
		j=1;
		for (i=n;i>=1;i--){
			res+=num[i]*j;
			j*=10;
		}
		return res;
	}
	
	public static void GetGrayScale(String s, int scale[][], int width, int height) throws IOException{
		int sl,value;
		BufferedInputStream f = new BufferedInputStream (new FileInputStream(s));
		sl=0;
		while (true){
			value = f.read();
			if ((value==10)||(value==32)){
				sl++;
				if (sl==4) break;
			}
		}
		for (int y=0;y<height;y++)
			for (int x=0;x<width;x++) scale[x][y]=f.read();
		f.close();
	}
	
	public static void Converter(String Origin, String Destination) throws IOException{
		int sl,n,value,width,height;
		int num[] = new int [10];
		BufferedInputStream f = new BufferedInputStream (new FileInputStream(Origin));
		width=0;
		height=0;
		n=0;
		sl=0;
		while (true){
			value = f.read();
			if ((value==10)||(value==32)){
				sl++;
				if (sl==1) n=0;
				if (sl==2){
					width=GetNumber(num,n);
					n=0;
				}
				if (sl==3){
					height=GetNumber(num,n);
					n=0;
				}
				if (sl==4) break;
			}else{
				n++;
				num[n]=value-(int)('0');
			}
		}
		String type = GetType(Destination);
		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		for (int y=0;y<height;y++)
			for (int x=0;x<width;x++){
				value=f.read();
				image.setRGB(x,y,RGB(value,value,value));
			}
        File file = new File(Destination);
		ImageIO.write(image, type, file);
		f.close();
	}
	
}
