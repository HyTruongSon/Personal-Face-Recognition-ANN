package MyLib;

import java.lang.*;
import java.io.*;

public class MedianFilter {

	static int width, height;
	static int a[] = new int [111];
	
	public static int FindMedian(int f[][],int x,int y){
		int i,j,u,v,n;
		int temp;
		n=0;
		for (i=-1;i<=1;i++)
			for (j=-1;j<=1;j++){
				u=x+i;
				v=y+j;
				if ((u>=0)&&(u<width)&&(v>=0)&&(v<height)){
						n++;
						a[n]=f[u][v];
					}
			}
		for (i=1;i<n;i++)
			for (j=i+1;j<=n;j++)
				if (a[i]>a[j]){
					temp = a[i];
					a[i] = a[j];
					a[j] = temp;
				}
		return a[(n+1)/2];
	}
	
	public static void process(int f[][],int g[][],int w,int h){
		int x,y;
		width=w;
		height=h;
		for (x=0;x<width;x++)
			for (y=0;y<height;y++)
				g[x][y]=FindMedian(f,x,y);
	}
	
	public static void process(int f[][],int w,int h){
		Runtime r = Runtime.getRuntime();
		r.gc();
		
		int x,y;
		int g[][] = new int [w][h];
		process(f,g,w,h);
		for (x=0;x<w;x++)
			for (y=0;y<h;y++) f[x][y]=g[x][y];
			
		r.gc();
	}

}