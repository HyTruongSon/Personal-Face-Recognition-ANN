package MyLib;

import java.lang.*;
import java.io.*;

public class Canny {

	static int[] DX = {0,0,-1,1,-1,-1,1,1};
	static int[] DY = {-1,1,0,0,-1,1,-1,1};
	static int[][] SmoothMask = { {2,4,5,4,2} , {4,9,12,9,4} , {5,12,15,12,5} , {4,9,12,9,4} , {2,4,5,4,2} };
	static int[][] SobelX = { {-1,0,1} , {-1,0,1} , {-1,0,1} };
	static int[][] SobelY = { {-1,-1,-1} , {0,0,0} , {1,1,1} };
	static int[][] PerwittX = { {-1,0,1} , {-2,0,2} , {-1,0,1} };
	static int[][] PerwittY = { {-1,-2,-1} , {0,0,0} , {1,2,1} };
	
	public static void process(int scale[][],int g[][],int width,int height,int LowThreshold,int HighThreshold){
		Runtime r = Runtime.getRuntime();
		r.gc();
	
		int i,j,x,y,u,v,x1,y1,x2,y2,gray;
		double gx,gy;
		boolean ok;
		int f[][];
		double gradient[][], anfa[][];
		boolean edge[][];
		
		f = new int [width][height];
		edge = new boolean [width][height];
		
		for (x=0;x<width;x++)
			for (y=0;y<height;y++){
				edge[x][y]=true;
				f[x][y]=0;
				for (i=-2;i<=2;i++)
					for (j=-2;j<=2;j++){
						u=x+i;
						v=y+j;
						if ((u>=0)&&(u<width)&&(v>=0)&&(v<height))
							f[x][y]+=scale[x+i][y+j]*SmoothMask[2+i][2+j];
					}
				f[x][y]/=159;
			}
			
		for (x=0;x<width;x++)
			for (y=0;y<height;y++) scale[x][y]=f[x][y];
			
		gradient = new double [width][height];
		anfa = new double [width][height];
		
		for (x=0;x<width;x++)
			for (y=0;y<height;y++){
				gx=0;
				gy=0;
				for (i=-1;i<=1;i++)
					for (j=-1;j<=1;j++){
						u=x+i;
						v=y+j;
						if ((u>=0)&&(u<width)&&(v>=0)&&(v<height)){
							gx+=scale[u][v]*SobelX[1+i][1+j];
							gy+=scale[u][v]*SobelY[1+i][1+j];
						}
					}
				gradient[x][y]=Math.sqrt(gx*gx+gy*gy);
				if (gx==0) anfa[x][y]=Math.PI/2; else
					anfa[x][y]=Math.atan(gy/gx);
			}
			
		for (x=1;x<width-1;x++)
			for (y=1;y<height-1;y++){
				x1=0;
				y1=0;
				x2=0;
				y2=0;
				if ((anfa[x][y]<=Math.PI/8)||(anfa[x][y]>7*Math.PI/8)){
					x1=x-1;
					y1=y;
					x2=x+1;
					y2=y;
				}
				if ((anfa[x][y]>Math.PI/8)&&(anfa[x][y]<=3*Math.PI/8)){
					x1=x-1;
					y1=y-1;
					x2=x+1;
					y2=y+1;
				}
				if ((anfa[x][y]>3*Math.PI/8)&&(anfa[x][y]<=5*Math.PI/8)){
					x1=x-1;
					y1=y-1;
					x2=x+1;
					y2=y-1;
				}
				if ((anfa[x][y]>5*Math.PI/8)&&(anfa[x][y]<=7*Math.PI/8)){
					x1=x;
					y1=y+1;
					x2=x;
					y2=y-1;
				}
				if ((gradient[x][y]<gradient[x1][y1])||(gradient[x][y]<gradient[x2][y2]))
					edge[x][y]=false;
			}
			
		for (x=1;x<width-1;x++)
			for (y=1;y<height-1;y++)
				if (edge[x][y]){
					if (gradient[x][y]>=HighThreshold) continue;
					if (gradient[x][y]<LowThreshold){
						edge[x][y]=false;
						continue;
					}
					ok=false;
					for (i=0;i<8;i++){
						u=x+DX[i];
						v=y+DY[i];
						if ((u>=0)&&(u<width)&&(v>=0)&&(v<height))
							if (gradient[u][v]>=HighThreshold){
								ok=true;
								break;
							}
					}
					if (ok) continue;
					edge[x][y]=false;
				}
				
		for (x=0;x<width;x++)
			for (y=0;y<height;y++)
				if (!edge[x][y]) gradient[x][y]=0; 
				
		for (x=0;x<width;x++)
			for (y=0;y<height;y++){
					gray=(int)(gradient[x][y]);
					if ((x==0)||(x==width-1)||(y<=2)||(y>=height-3)) gray=0;
					g[x][y]=gray;
				}
				
		r.gc();
	}
	
	public static void process(int scale[][],int width,int height,int LowThreshold,int HighThreshold){
		Runtime r = Runtime.getRuntime();
		r.gc();
	
		int x,y;
		int g[][] = new int [width][height];
		process(scale,g,width,height,LowThreshold,HighThreshold);
		for (x=0;x<width;x++)
			for (y=0;y<height;y++) scale[x][y]=g[x][y];
			
		r.gc();
	}
	
}