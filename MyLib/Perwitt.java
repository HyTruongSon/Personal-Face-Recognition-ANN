package MyLib;

import java.lang.*;
import java.io.*;

public class Perwitt {
	
	static int width, height;
	static int[][] PerwittX = { {-1,0,1} , {-2,0,2} , {-1,0,1} };
	static int[][] PerwittY = { {-1,-2,-1} , {0,0,0} , {1,2,1} };
	
	public static int PerwittGradient(int f[][],int x,int y){
		int i,j,gx,gy;
		if ((x==0)||(x==width-1)||(y==0)||(y==height-1)) return 0;
		gx=0;
		gy=0;
		for (i=-1;i<=1;i++)
			for (j=-1;j<=1;j++){
				gx+=f[x-i][y-j]*PerwittX[1-i][1-j];
				gy+=f[x-i][y-j]*PerwittY[1-i][1-j];
			}
		return Math.abs(gx)+Math.abs(gy);
	}
	
	public static void process(int f[][],int g[][],int w,int h){
		int x,y;
		width = w;
		height = h;
		for (x=0;x<width;x++)
			for (y=0;y<height;y++)
				g[x][y]=PerwittGradient(f,x,y);
	}
	
	public static void process(int f[][],int w,int h){
		Runtime r = Runtime.getRuntime();
		r.gc();
		
		int x,y;
		width = w;
		height = h;
		int g[][] = new int [width][height];
		process(f,g,width,height);
		for (x=0;x<width;x++)
			for (y=0;y<height;y++) f[x][y]=g[x][y];
			
		r.gc();
	}

}